/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/arch/arch.h"

#include <cooperative_groups.h>
#include <cmath>
#include "cutlass/gemm_ring_queue.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cooperative_groups;

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct Gemm {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int *semaphore;
    int gemm_k_size;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr,
      int const *gather_A_indices = nullptr,
      int const *gather_B_indices = nullptr,
      int const *scatter_D_indices = nullptr
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op),
      gather_A_indices(gather_A_indices),
      gather_B_indices(gather_B_indices),
      scatter_D_indices(scatter_D_indices) {

      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      
      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

    semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Gemm() { 
    // unsigned int smid;
    // asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    // if(threadIdx.x==0 && smid == 0){
    //   printf("smid: %d\n", smid);
    // }
  } 

  /// Determines whether kernel satisfies alignment
  CUTLASS_HOST_DEVICE
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size,
    typename Mma::IteratorA::TensorRef ref_A,
    typename Mma::IteratorB::TensorRef ref_B,
    typename Epilogue::OutputTileIterator::TensorRef ref_C,
    typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =  (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;
    
    // printf("%d, %d, %d \n", kAlignmentA, kAlignmentB, kAlignmentC);
    // (4,4,4)

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

__device__ void block_to_coordinate(int block_idx, int grid_tiled_shape_m,
                                    int &threadblock_tile_offset_m, int &threadblock_tile_offset_n){
  
  threadblock_tile_offset_m = block_idx % grid_tiled_shape_m;
  threadblock_tile_offset_n = block_idx / grid_tiled_shape_m;
}

__device__ int get_corresponding_chk_idx(int grid_tiled_shape_m, int matrix_blk, 
                                        int threadblock_tile_offset_m, int matrix_shape_m){
  int n = (matrix_blk) / grid_tiled_shape_m;
  int m = threadblock_tile_offset_m / 128;
  // int chk_blk = grid_tiled_shape_m * (n + 1) - 1;
  int chk_blk = (grid_tiled_shape_m * n) + matrix_shape_m + m;
  return chk_blk;
}

__device__ void SM_based_schedule(Params const &params, int threadblock_tile_offset_m, int threadblock_tile_offset_n,
                                  int &tmp_matrix_blk, int &tmp_chk_blk, int &tmp_flag,
                                  unsigned int smid, int block_idx, int matrix_SM, int iter, int checksumblk_per_col){
  if (smid < matrix_SM){
    // issue there
    // tmp_matrix_blk = (block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
    
    int new_blk_idx = block_idx - threadblock_tile_offset_n * checksumblk_per_col;
    int group_idx = new_blk_idx / matrix_SM;

    int num_group = (params.grid_tiled_shape.m() - checksumblk_per_col) * params.grid_tiled_shape.n() / matrix_SM;
    int remaining_blk = (params.grid_tiled_shape.m() - checksumblk_per_col) * params.grid_tiled_shape.n() % matrix_SM;
    int previous_blk_size = matrix_SM;
    if(remaining_blk == 1){
      if(group_idx == (num_group-2)){
        matrix_SM++;
      }
      if(group_idx == (num_group-1)){
        group_idx--;
        matrix_SM++;
      }
    }
    else if(remaining_blk > 1){
      if(group_idx == (num_group-1)){
        matrix_SM = remaining_blk;
      }
    }

    int local_blk_idx = new_blk_idx % matrix_SM;
    int next_local_blk_idx = (local_blk_idx + 1) % matrix_SM;
    int next_global_blk_idx = next_local_blk_idx + (group_idx * previous_blk_size);
    int new_offset_n = (next_global_blk_idx / (params.grid_tiled_shape.m() - checksumblk_per_col)) * checksumblk_per_col;
    tmp_matrix_blk = next_global_blk_idx + new_offset_n;

    // if(iter==348){
    //   printf("1 Check %dth. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix block: (%d), next chk block: (%d)\n", 
    //           iter, block_idx, threadblock_tile_offset_m, threadblock_tile_offset_n, smid, tmp_matrix_blk, tmp_chk_blk);
    // }
    
    // if ((tmp_matrix_blk + 1) % (params.grid_tiled_shape.m()-(checksumblk_per_col-1)) == 0){
    //   tmp_matrix_blk = (tmp_matrix_blk + checksumblk_per_col) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
    // }
    // int n = (tmp_matrix_blk) / params.grid_tiled_shape.m();
    // int m = threadblock_tile_offset_m / 128;
    // tmp_chk_blk = params.grid_tiled_shape.m() * (n + 1) - 1;
    
    tmp_chk_blk = get_corresponding_chk_idx(params.grid_tiled_shape.m(), tmp_matrix_blk, threadblock_tile_offset_m, (params.grid_tiled_shape.m() - checksumblk_per_col));
    tmp_flag = 1;

    // if(iter==348){
      // printf("Check %dth. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix block: (%d), next chk block: (%d)\n", 
      //         iter, block_idx, threadblock_tile_offset_m, threadblock_tile_offset_n, smid, tmp_matrix_blk, tmp_chk_blk);
    // }

  }
  else{
  // printf("chksum. block_idx: %d, tile_offset.m: %d, title_offset.n: %d, SM: %d, \n", 
  //     block_idx, threadblock_tile_offset_m, threadblock_tile_offset_n, smid);
  }
}

  __device__ void curr_iter_chk_offsets(Params const &params, int &matrix_start_idx, int &chk_start_idx,
                                        int next_matrix_block_idx, int next_chk_block_idx, int checksumblk_per_col, 
                                        int thread_idx){
    int offset = blockDim.x;

    int MatrixColBlkOffset = next_matrix_block_idx / params.grid_tiled_shape.m();
    int MatrixRowBlkOffset = next_matrix_block_idx % params.grid_tiled_shape.m();
    matrix_start_idx = (MatrixColBlkOffset * offset) + (MatrixRowBlkOffset * 128) * params.problem_size.n() + thread_idx;

    int ChkColBlkOffset = next_chk_block_idx / params.grid_tiled_shape.m();
    int ChkRowBlkOffset = (params.grid_tiled_shape.m() - checksumblk_per_col);
    chk_start_idx = (ChkColBlkOffset * offset) + (ChkRowBlkOffset * 128 + 1 * MatrixRowBlkOffset) * params.problem_size.n() + thread_idx;
  }


  __device__ void last_iter_chk_offsets(Params const &params, int &matrix_start_idx, int &chk_start_idx,
                                        int next_matrix_block_idx, int next_chk_block_idx, 
                                        int checksumblk_per_col, int matrix_next_blk_offset_m, int matrix_next_blk_offset_n, 
                                        int thread_idx){
    int add_col = 0;
    int offset = blockDim.x;
    
    int MatrixRowBlkOffset = next_matrix_block_idx % params.grid_tiled_shape.m() - matrix_next_blk_offset_m;
    if(MatrixRowBlkOffset < 0){
      MatrixRowBlkOffset += (params.grid_tiled_shape.m() - checksumblk_per_col);
      add_col = 1;
    }
    int MatrixColBlkOffset = next_matrix_block_idx / params.grid_tiled_shape.m() - matrix_next_blk_offset_n - add_col;
    matrix_start_idx = (MatrixColBlkOffset * offset) + (MatrixRowBlkOffset * 128) * params.problem_size.n() + thread_idx;

    int ChkRowBlkOffset = (params.grid_tiled_shape.m() - checksumblk_per_col);
    int ChkColBlkOffset = next_chk_block_idx / params.grid_tiled_shape.m() - matrix_next_blk_offset_n - add_col;
    chk_start_idx = (ChkColBlkOffset * offset) + (ChkRowBlkOffset * 128 + 1 * MatrixRowBlkOffset) * params.problem_size.n() + thread_idx;
  }


  __device__ void last_iter_chk_offsets_v2(Params const &params, int &matrix_start_idx, int &chk_start_idx,
                                        int next_matrix_block_idx, int checksumblk_per_col, int matrix_SM, int thread_idx){
    // matrix_blk
    int last_local_blk = next_matrix_block_idx - (next_matrix_block_idx / params.grid_tiled_shape.m()) * checksumblk_per_col - matrix_SM;
    int last_matrix_block_idx = last_local_blk + (last_local_blk / (params.grid_tiled_shape.m() - checksumblk_per_col)) * checksumblk_per_col;
    int last_threadblock_tile_offset_m = last_matrix_block_idx % params.grid_tiled_shape.m();
    
    // check blk
    int last_chk_block_idx = get_corresponding_chk_idx(params.grid_tiled_shape.m(), last_matrix_block_idx, last_threadblock_tile_offset_m, (params.grid_tiled_shape.m() - checksumblk_per_col));
  
    // offsets
    curr_iter_chk_offsets(params, matrix_start_idx, chk_start_idx, last_matrix_block_idx, last_chk_block_idx, checksumblk_per_col, thread_idx);
  }


  __device__ void check_phase(Params const &params, int matrix_start_idx, int chk_start_idx, int *SM_check_res, 
                              int iter, 
                              // int *recompute, int *compare, int *checking, 
                              unsigned int smid, int thread_idx, int next_matrix_block_idx, int next_chk_block_idx, int block_idx){
    float recomputed_chksum = 0;
    int diff = 0;
    
    // if use group, not unroll
    int N = params.problem_size.n();
    // void *p = params.ref_D.data();
    #pragma unroll
    for(int r = 0; r < 128; r++){
      int idx = matrix_start_idx + r * N;
      recomputed_chksum += *(params.ref_D.data() + idx);
      // float temp = params.ref_D.data(idx);
    }
    
    // __syncthreads();
    // if(thread_idx == 0 && smid == 0){
    //   *(recompute + iter) = clock();
    // }
    
    if(fabs(recomputed_chksum - (*(params.ref_D.data() + chk_start_idx))) > (float)1e3){
      diff = 1;
      printf("%d Difference detected at (%d, %d, %d). next matrix sum: (%d, %f), next chk: (%d, %f)\n", 
                iter, smid, block_idx, thread_idx, next_matrix_block_idx, recomputed_chksum, next_chk_block_idx, *(params.ref_D.data() + chk_start_idx));
    }
    // __syncthreads();
    // if(thread_idx == 0 && smid == 0){
    //   *(compare + iter) = clock();
    // }

    // Atomic sum
    if(diff != 0){
      atomicAdd((SM_check_res+smid), diff);
    }
    __syncthreads();
    if(*(SM_check_res+smid)!=0){
      if(thread_idx == 0){
        // printf("%d,  Difference detected at SM %d. Reduced Sum: %d\n", iter, smid, *(SM_check_res+smid));
        // *(SM_check_res+smid) = 0;
      }
    }

    // __syncthreads();
    // if(thread_idx == 0 && smid == 0){
    //   *(checking + iter) = clock();
    //   // printf("checking: %d\n", *(checking + iter));
    // }
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, 
                  int if_split_phase, int *SM_check_res, int partion
                  // int *all_start, int *compute, int *finding, int *recompute, int *compare, int *checking
                ) {

    // get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    int threadblock_tile_offset_m, threadblock_tile_offset_k, threadblock_tile_offset_n;

    // SM based schedule info
    int checksumblk_per_col = 0;
    if(if_split_phase == 0){
      // if able ABFT
      // checksumblk_per_col = (int)(ceil((double)((params.grid_tiled_shape.m()) / (double)(128))));
      checksumblk_per_col = (int)(ceil((double)((partion) / (double)(128))));
    }
    
    int matrix_shape_m = params.grid_tiled_shape.m() - checksumblk_per_col;

    int max_col = (int)ceil((double)132 / (double)(matrix_shape_m));
    if(max_col > params.grid_tiled_shape.n()){
      max_col = params.grid_tiled_shape.n();
    }

    int remaining_SM = (int)(max_col * checksumblk_per_col);
    int matrix_SM = (int)(132 - remaining_SM);

    int matrix_next_blk_offset_m = matrix_SM % (matrix_shape_m);
    int matrix_next_blk_offset_n = (matrix_SM / matrix_shape_m);
    int checksum_next_blk_offset_n = (checksumblk_per_col != 0) ? (remaining_SM / checksumblk_per_col) : 0;
    // iteration based on GeMM not (GeMM + chksum)
    int SM_iter = (int)ceil((double)((matrix_shape_m * params.grid_tiled_shape.n())/(double)matrix_SM));
              
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    // if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
    //   params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

    //   return;
    // }

    for(int iter = 0; iter < SM_iter; iter++){

    bool beyond_bound = false;
    threadblock_tile_offset_k = threadblock_tile_offset.k();
    // int block_idx;

    // new offset - first Z matrix, last Y checksum
    // SM_schedule (num of SM for matrix, num of SM of chk, chk blk row, matrix offset_m, matrix offset_n, chk offset_n)

    if(smid < matrix_SM){
      // for matrix
      // int local_matrix_idx = smid + iter * matrix_SM;
      // block_idx = local_matrix_idx + (local_matrix_idx / matrix_shape_m) * checksumblk_per_col;
      // block_to_coordinate(block_idx, params.grid_tiled_shape.m(), threadblock_tile_offset_m, threadblock_tile_offset_n);
      
      int add_col = 0;
      threadblock_tile_offset_m = (smid % (matrix_shape_m) + iter * matrix_next_blk_offset_m) % (matrix_shape_m);
      add_col = ((smid % (matrix_shape_m) + iter * matrix_next_blk_offset_m) / (matrix_shape_m));
      threadblock_tile_offset_n = smid / (matrix_shape_m) + iter * matrix_next_blk_offset_n + add_col;
    }
    else{
      // for checksum
      // unsigned int local_chk_blk_idx = (smid - matrix_SM) + iter * remaining_SM;
      // block_to_coordinate(local_chk_blk_idx, checksumblk_per_col, threadblock_tile_offset_m, threadblock_tile_offset_n);
      // threadblock_tile_offset_m += matrix_shape_m;
      // block_idx = threadblock_tile_offset_m + threadblock_tile_offset_n * params.grid_tiled_shape.m();
      
      unsigned int local_chk_blk_idx = smid - matrix_SM;
      threadblock_tile_offset_m = matrix_shape_m + (local_chk_blk_idx % checksumblk_per_col);
      threadblock_tile_offset_n = local_chk_blk_idx / checksumblk_per_col + iter * checksum_next_blk_offset_n;
    }
    if(threadblock_tile_offset_n >= params.grid_tiled_shape.n()){
      // return;
      beyond_bound = true;
    }
    
    // if(smid == 23 && threadIdx.x==0){
    //   printf("smid: %d, m: %d, n:%d\n", smid, threadblock_tile_offset_m,threadblock_tile_offset_n);
    // }
    
    int block_idx = threadblock_tile_offset_m + threadblock_tile_offset_n * params.grid_tiled_shape.m();
    int thread_idx = threadIdx.x;

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if(!beyond_bound){
    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset_m * Mma::Shape::kM,
      threadblock_tile_offset_k * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset_k * params.gemm_k_size,
      threadblock_tile_offset_n * Mma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(), 
      (threadblock_tile_offset_k + 1) * params.gemm_k_size);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute position within threadblock
    // int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      params.gather_A_indices);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      params.gather_B_indices);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();
    
    // get num of SM
    // unsigned int nsm;
    // asm volatile("mov.u32  %0, %nsmid;" : "=r"(nsm));
    // printf("Num of SM: %d\n", nsm);
    
    // get thread id in each SM
    // unsigned int tid;
    // asm volatile("mov.u32 %0,%tid.x;" : "=r"(tid));
    // printf("SM id: %d, tid: %d, thread_idx;%d\n", smid, tid, thread_idx);
    
    // printf("M: %d, N: %d, K: %d \n", params.problem_size.m(), problem_size_k, params.problem_size.n());

    // __syncthreads();
    // if(thread_idx == 0 && smid == 0){
    //   *(all_start + iter) = clock();
    //   // printf("all_start: %d\n", *all_start);
    // }

    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset_m * Mma::Shape::kM,
      threadblock_tile_offset_n * Mma::Shape::kN
    );

    // printf("SM id: %d, A_data: %f, A_row_offset: %d, A_col_offset: %d, B_data: %f, B_row_offset: %d, B_col_offset: %d, C_row_offset: %d, C_col_offset: %d\n", 
    //   smid, *(params.ref_A.data()+off_A), tb_offset_A.row(), tb_offset_A.column(), 
    //         *(params.ref_B.data()+off_B), tb_offset_B.row(), tb_offset_B.column(),
    //         threadblock_offset.row(), threadblock_offset.column());

    // int block_idx = threadblock_tile_offset_m + threadblock_tile_offset_n * params.grid_tiled_shape.m();
    
    // __syncthreads();
    // if(smid == 1 && thread_idx == 0){
    //   printf("accumulators:\n");
    //   for(int r = 0; r < 1; r++){
    //     for(int c = 0; c < 128; c++){
    //       int idx = 0 + (r + c * 128);
    //       // if(*(accumulators.data() + idx) != (float)12800.000000){
    //         printf("(%d, %d, %f); ", iter, block_idx, *(accumulators.data() + idx));
    //       // }
    //     }
    //     printf("\n");
    //   }
    // }
    
    // if(threadIdx.x == 0){
    //   printf("SM id: %d, block id: %d\n", smid, block_idx);
    // }

    // printf("block_idx: %d, tile_offset.m: %d, title_offset.n: %d, grid_tile_shape.m: %d, grid_tile_shape.n: %d\n", 
    //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), params.grid_tiled_shape.m(), params.grid_tiled_shape.n());

    // Construct the semaphore.
    // Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset_k, params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset_k) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset_k);

    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C); 

    // Simulate compute matrix error
    // if(smid == 0){
    //   if(thread_idx == 0){
    //     int inject_idx = threadblock_offset.column() + threadblock_offset.row() * params.problem_size.n();

    //     *(params.ref_D.data() + inject_idx) = 0;
    //   }
    //   // __syncthreads();
    // }

    // int off_A = tb_offset_A.row() + tb_offset_A.column()*params.problem_size.m();
    // int off_B = tb_offset_B.row() + tb_offset_B.column()*params.problem_size.n();
    // int off_C = threadblock_offset.row() + threadblock_offset.column() * params.problem_size.m();

    // int off_A = tb_offset_A.column() + tb_offset_A.row()*params.problem_size.k();
    // int off_B = tb_offset_B.column() + tb_offset_B.row()*params.problem_size.n();
    // int off_C = threadblock_offset.column() + threadblock_offset.row() * params.problem_size.n();

    // printf("SM id: %d, Block idx: %d, A_data: %f, A_row_offset: %d, A_col_offset: %d, B_data: %f, B_row_offset: %d, B_col_offset: %d, D_data: %f, D_row_offset: %d, D_col_offset: %d\n", 
    //           smid, block_idx, *(params.ref_A.data()+off_A), tb_offset_A.row(), tb_offset_A.column(), 
    //                           *(params.ref_B.data()+off_B), tb_offset_B.row(), tb_offset_B.column(),
    //                           *(params.ref_D.data()+off_C), threadblock_offset.row(), threadblock_offset.column());

    // if(block_idx == 1 && thread_idx == 0){
    //   printf("A:\n");
    //   for(int r = 0; r < 1; r++){
    //     for(int c = 0; c < 256; c++){
    //       int idx = 128*256 + (c + r * params.problem_size.k());
    //       printf("%f, ", *(params.ref_A.data() + idx));
    //     }
    //     printf("\n");
    //   }
    // }

    // 
    // Signature and Find (Matrix SM) (Column Checksum Only)
    //
    // __shared__ unsigned int next_chk_smid, next_matrix_smid;
    // __syncthreads();
    // if(thread_idx == 0 && smid == 0){
    //   *(compute + iter) = clock();
    // }
    }
    // if(block_idx == 94){
    //   printf("iter: %d, thread: %d, value: %f\n", iter, thread_idx, *(params.ref_D.data() + thread_idx));
    // }
    
    // if(iter == 0 && (*((SM_schedule)+6)) != 1){
    //   continue;
    // }
    // else if(iter == (*((SM_schedule)+6))-1){
    //   cooperative_groups::this_grid().sync();
    // }
    
    // __syncthreads();

    // if(beyond_bound){
    //   return;
    // }
    #if 1
    if(if_split_phase == 0){
      if(iter == 0 && SM_iter != 1){
        continue;
      }
      else if(iter == (SM_iter - 1)){
        cooperative_groups::this_grid().sync();
      }

      // __shared__ int next_matrix_block_idx, next_chk_block_idx, flag;
      int *int_smem = reinterpret_cast<int *>(&shared_storage);
      int &next_matrix_block_idx = int_smem[0];
      int &next_chk_block_idx = int_smem[1];
      int &flag = int_smem[2];

      int tmp_matrix_blk, tmp_chk_blk, tmp_flag;
      // int matrix_SM = *(SM_schedule);
      // block view
      if(thread_idx == 0){
        SM_based_schedule(params, threadblock_tile_offset_m, threadblock_tile_offset_n, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx, matrix_SM, iter, checksumblk_per_col);

        next_matrix_block_idx = tmp_matrix_blk;
        // 
        next_chk_block_idx = tmp_chk_blk;
        // 
        flag = tmp_flag;
      }
      __syncthreads();
      // if(thread_idx == 0 && smid == 0){
      //   *(finding + iter) = clock();
      // }

      // begin chkeck
      if(flag == 1){
        if (smid < matrix_SM){
          int matrix_start_idx, chk_start_idx;
          // iter 1 ~ (n-2)
          if(iter < (SM_iter - 1) && iter > 0){
            last_iter_chk_offsets(params, matrix_start_idx, chk_start_idx, next_matrix_block_idx, next_chk_block_idx, checksumblk_per_col, matrix_next_blk_offset_m, matrix_next_blk_offset_n, thread_idx);
            
            // int a, b;
            // last_iter_chk_offsets_v2(params, matrix_start_idx, chk_start_idx, next_matrix_block_idx, checksumblk_per_col, matrix_SM, thread_idx);
            // if(thread_idx == 0){
            //   if((matrix_start_idx != a) || (chk_start_idx != b)){
            //     printf("iter: %d, matrix: (%d, %d), chksum: (%d, %d)\n", iter, matrix_start_idx, a, chk_start_idx, b);
            //   }
            // }

            // check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, iter, recompute, compare, checking, smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx);
            check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, iter, smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx);
          }
          // iter n-1
          else if(iter == SM_iter - 1){
            /*
            2 cases: 
              1. only one iter: 
                matrix & chk offset
                recompute & compare & atomic
                exit
              2. last iter, more than one iters:
                2 steps:
                  check last
                  check self
            */
            int ti = iter;
            if(SM_iter != 1){
              //check last iteration
              last_iter_chk_offsets(params, matrix_start_idx, chk_start_idx, next_matrix_block_idx, next_chk_block_idx, checksumblk_per_col, matrix_next_blk_offset_m, matrix_next_blk_offset_n, thread_idx);
              // check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, ti, recompute, compare, checking, smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx);
              
              // int a, b;
              // last_iter_chk_offsets_v2(params, matrix_start_idx, chk_start_idx, next_matrix_block_idx, checksumblk_per_col, matrix_SM, thread_idx);

              // if(thread_idx == 0){
              //   if((matrix_start_idx != a) || (chk_start_idx != b)){
              //     printf("iter: %d, (%d, %d), (%d, %d), matrix: (%d, %d), chksum: (%d, %d)\n", 
              //             iter, threadblock_tile_offset_m, threadblock_tile_offset_n, block_idx, next_matrix_block_idx, matrix_start_idx, a, chk_start_idx, b);
              //   }
              // } 

              check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, ti, smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx);
              ti++;
            }
            // cooperative_groups::this_grid().sync();
            if(beyond_bound){
              return;
            }
            // check current iteration
            curr_iter_chk_offsets(params, matrix_start_idx, chk_start_idx, next_matrix_block_idx, next_chk_block_idx, checksumblk_per_col, thread_idx);
            // check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, ti, recompute, compare, checking, smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx);
            check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, ti, smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx);
          }
        }
      }
    }
    // else if(if_split_phase == 1 && block_idx == 0){
    //   // 
    //   *(Signature_Array + block_idx) = (uint8_t)smid;
    // }
    else{
      // __syncthreads();
      // if(thread_idx == 0 && smid == 0){
      //   *(checking + iter) = clock();
      // }
    }
    #endif
        
    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset_k + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset_k + 1;
      }

      semaphore.release(lock);
    }
  }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass