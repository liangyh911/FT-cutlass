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

#include <cooperative_groups.h>

using namespace cooperative_groups;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmBatched {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    int swizzle_log_tile{0};
    typename Mma::IteratorA::Params params_A{};
    typename Mma::IteratorA::TensorRef ref_A{};
    int64_t stride_A{0};
    typename Mma::IteratorB::Params params_B{};
    typename Mma::IteratorB::TensorRef ref_B{};
    int64_t stride_B{0};
    typename Epilogue::OutputTileIterator::Params params_C{};
    typename Epilogue::OutputTileIterator::TensorRef ref_C{};
    int64_t stride_C{0};
    typename Epilogue::OutputTileIterator::Params params_D{};
    typename Epilogue::OutputTileIterator::TensorRef ref_D{};
    int64_t stride_D{0};
    typename OutputOp::Params epilogue{};
    int batch_count{1};
    int gemm_k_iterations{0};

    //
    // Methods
    //
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size_,
      cutlass::gemm::GemmCoord const & grid_tiled_shape_,
      typename Mma::IteratorA::TensorRef ref_A_,
      int64_t stride_A_,
      typename Mma::IteratorB::TensorRef ref_B_,
      int64_t stride_B_,
      typename Epilogue::OutputTileIterator::TensorRef ref_C_,
      int64_t stride_C_,
      typename Epilogue::OutputTileIterator::TensorRef ref_D_,
      int64_t stride_D_,
      typename OutputOp::Params epilogue_,
      int batch_count_
    ):
      problem_size(problem_size_),
      grid_tiled_shape(grid_tiled_shape_),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A_.layout()),
      ref_A(ref_A_),
      stride_A(stride_A_),
      params_B(ref_B_.layout()),
      ref_B(ref_B_),
      stride_B(stride_B_),
      params_C(ref_C_.layout()),
      ref_C(ref_C_),
      stride_C(stride_C_),
      params_D(ref_D_.layout()),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_),
      gemm_k_iterations((problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK) {}
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

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
      
      tmp_chk_blk = get_corresponding_chk_idx(params.grid_tiled_shape.m(), tmp_matrix_blk, threadblock_tile_offset_m, (params.grid_tiled_shape.m() - checksumblk_per_col));
      tmp_flag = 1;
     
      // printf("Check %dth. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix block: (%d), next chk block: (%d)\n", 
      //         iter, block_idx, threadblock_tile_offset_m, threadblock_tile_offset_n, smid, tmp_matrix_blk, tmp_chk_blk);
    }
    else{
    // printf("chksum. block_idx: %d, tile_offset.m: %d, title_offset.n: %d, SM: %d, \n", 
    //     block_idx, threadblock_tile_offset_m, threadblock_tile_offset_n, smid);
    }
  }

  __device__ void next_blks_idx(Params const &params, int threadblock_tile_offset_m,
    int &tmp_matrix_blk, int &tmp_chk_blk, int &tmp_flag,
    unsigned int smid, int block_idx, int matrix_SM, int iter, int checksumblk_per_col){

    tmp_matrix_blk = (block_idx + 1) % matrix_SM;
    tmp_chk_blk = get_corresponding_chk_idx(params.grid_tiled_shape.m(), tmp_matrix_blk, threadblock_tile_offset_m, params.grid_tiled_shape.m());
    tmp_flag = 1;
    }

  __device__ void previous_blks_idx(Params const &params, int threadblock_tile_offset_m,
    int &tmp_matrix_blk, int &tmp_chk_blk,
    unsigned int smid, int block_idx, int matrix_SM, int iter, int checksumblk_per_col){

    tmp_matrix_blk = block_idx - 1;
    if(tmp_matrix_blk < 0){
      tmp_matrix_blk = matrix_SM + tmp_matrix_blk;
    }
    int threadblock_tile_offset_n = tmp_matrix_blk / params.grid_tiled_shape.m();
    tmp_matrix_blk = tmp_matrix_blk + threadblock_tile_offset_n * checksumblk_per_col;

    tmp_chk_blk = get_corresponding_chk_idx((params.grid_tiled_shape.m() + checksumblk_per_col), tmp_matrix_blk, threadblock_tile_offset_m, params.grid_tiled_shape.m());
  }

  __device__ void update_col(Params const &params, int matrix_blk, int chk_blk, int threadblock_tile_offset_n, int batch_idx, int thread_idx, int checksumblk_per_col){

    int M = params.problem_size.m();
    int K = params.problem_size.k();
    int N = params.problem_size.n();
    int gridM = params.grid_tiled_shape.m() + checksumblk_per_col;
    int tiled_N = blockDim.x; 

    int MatrixColBlkOffset = matrix_blk / gridM;
    int MatrixRowBlkOffset = matrix_blk % gridM;

    int col_idx = (MatrixColBlkOffset * tiled_N) + thread_idx;
  
    if(col_idx < N){
      float accum = 0.f;
      int idx_a = (batch_idx * params.stride_A) + (M + 1 * MatrixRowBlkOffset) * K;
      int idx_b = (batch_idx * params.stride_B) + (MatrixColBlkOffset * tiled_N) + thread_idx;
      int idx_chk = (batch_idx * params.stride_D) + (MatrixColBlkOffset * tiled_N) + (M + 1 * MatrixRowBlkOffset) * N + thread_idx;
  
      for(int k = 0; k < K; k++){
        float a = *(params.ref_A.data() + idx_a + k);
        float b = *(params.ref_B.data() + idx_b + k * N);
        accum += a * b;
      }
      *(params.ref_D.data() + idx_chk) = accum;
    }
  }
  
  __device__ void update_col_v2(Params const &params, float *smem, int threadblock_tile_offset_n, int batch_idx, int thread_idx, int checksumblk_per_col){

    int M = params.problem_size.m();
    int K = params.problem_size.k();
    int N = params.problem_size.n();
    int gridM = params.grid_tiled_shape.m() + checksumblk_per_col;
    int tiled_N = blockDim.x;

    int matrix_blk = (int)(*smem);
    int MatrixColBlkOffset = matrix_blk / gridM;
    int MatrixRowBlkOffset = matrix_blk % gridM;

    int col_idx = (MatrixColBlkOffset * tiled_N) + thread_idx; 
    
    // load to share memory (smem size: 73728, 18432, 72*256, 18*1024)
    float *d_A = smem + 2; 
    // float *d_B = smem + (2 + (1 * K));

    if(col_idx < K){
      int idx_a = (batch_idx * params.stride_A) + (M + 1 * MatrixRowBlkOffset) * K;
      int idx_da = col_idx;
      // int idx_b = (batch_idx * params.stride_B) + (MatrixColBlkOffset * tiled_N) + thread_idx;
      for(int k = 0; k < K; k++){
        *(d_A + idx_da + k * K) = *(params.ref_A.data() + idx_a + k * K);
        // *(d_B + k + K * thread_idx) = *(params.ref_B.data() + idx_b + k * N);
      }
    }
    __syncthreads();

  
    if(col_idx < N){
      float accum = 0.f;
      int idx_a = (1 * MatrixRowBlkOffset) * K;
      int idx_b = (batch_idx * params.stride_B) + (MatrixColBlkOffset * tiled_N) + thread_idx;
      int idx_chk = (batch_idx * params.stride_D) + (MatrixColBlkOffset * tiled_N) + (M + 1 * MatrixRowBlkOffset) * N + thread_idx;
  
      for(int k = 0; k < K; k++){
        // float a = *(params.ref_A.data() + idx_a + k);
        float a = *(d_A + idx_a + k);
        float b = *(params.ref_B.data() + idx_b + k * N);
        // if(thread_idx == 0) printf("a: %f, b: %f\n", a, b);
        accum += a * b;
      }
      *(params.ref_D.data() + idx_chk) = accum;
    }
  }

  __device__ void curr_iter_chk_offsets(Params const &params, int &matrix_start_idx, int &chk_start_idx,
                                        int next_matrix_block_idx, int next_chk_block_idx, int checksumblk_per_col, 
                                        int thread_idx, int batch_idx){
    int offset = blockDim.x;

    int MatrixColBlkOffset = next_matrix_block_idx / params.grid_tiled_shape.m();
    int MatrixRowBlkOffset = next_matrix_block_idx % params.grid_tiled_shape.m();
    matrix_start_idx = (batch_idx * params.stride_D) + (MatrixColBlkOffset * offset) + (MatrixRowBlkOffset * 128) * params.problem_size.n() + thread_idx;

    int ChkColBlkOffset = next_chk_block_idx / params.grid_tiled_shape.m();
    int ChkRowBlkOffset = (params.grid_tiled_shape.m() - checksumblk_per_col);
    chk_start_idx = (batch_idx * params.stride_D) + (ChkColBlkOffset * offset) + (ChkRowBlkOffset * 128 + 1 * MatrixRowBlkOffset) * params.problem_size.n() + thread_idx;
  }

  __device__ void last_iter_chk_offsets(Params const &params, int &matrix_start_idx, int &chk_start_idx,
                                        int next_matrix_block_idx, int next_chk_block_idx, 
                                        int checksumblk_per_col, int matrix_next_blk_offset_m, int matrix_next_blk_offset_n, 
                                        int thread_idx, int batch_idx){
    int add_col = 0;
    int offset = blockDim.x;
    
    int MatrixRowBlkOffset = next_matrix_block_idx % params.grid_tiled_shape.m() - matrix_next_blk_offset_m;
    if(MatrixRowBlkOffset < 0){
      MatrixRowBlkOffset += (params.grid_tiled_shape.m() - checksumblk_per_col);
      add_col = 1;
    }
    int MatrixColBlkOffset = next_matrix_block_idx / params.grid_tiled_shape.m() - matrix_next_blk_offset_n - add_col;
    matrix_start_idx = ((batch_idx-1) * params.stride_D) + (MatrixColBlkOffset * offset) + (MatrixRowBlkOffset * 128) * params.problem_size.n() + thread_idx;

    int ChkRowBlkOffset = (params.grid_tiled_shape.m() - checksumblk_per_col);
    int ChkColBlkOffset = next_chk_block_idx / params.grid_tiled_shape.m() - matrix_next_blk_offset_n - add_col;
    chk_start_idx = ((batch_idx-1) * params.stride_D) + (ChkColBlkOffset * offset) + (ChkRowBlkOffset * 128 + 1 * MatrixRowBlkOffset) * params.problem_size.n() + thread_idx;
  }

  __device__ void check_phase(Params const &params, int matrix_start_idx, int chk_start_idx, int *SM_check_res, 
                              int iter, 
                              unsigned int smid, int thread_idx, int next_matrix_block_idx, int next_chk_block_idx, 
                              int block_idx, int batch_idx, int threadblock_tile_offset_n){
    float recomputed_chksum = 0;
    int diff = 0;
    int col_idx = ((next_matrix_block_idx / params.grid_tiled_shape.m()) * blockDim.x) + threadIdx.x; 
    // int col_idx = (threadblock_tile_offset_n * blockDim.x) + threadIdx.x; 

    if(col_idx < params.problem_size.n()){
      int N = params.problem_size.n();
      // void *p = params.ref_D.data();
      #pragma unroll
      for(int r = 0; r < 128; r++){
        int idx = matrix_start_idx + r * N;
        recomputed_chksum += *(params.ref_D.data() + idx);
      }
      
      float check_sum_val = (*(params.ref_D.data() + chk_start_idx));
      if(fabs(recomputed_chksum - check_sum_val) > (float)1e1){
        diff = 1;
        printf("%d %d Difference detected at ((%d), %d, %d). next matrix sum: (%d, %f), next chk: (%d, %f), diff: %f\n", 
                  iter, batch_idx, smid, block_idx, thread_idx, next_matrix_block_idx, recomputed_chksum, next_chk_block_idx, *(params.ref_D.data() + chk_start_idx), (recomputed_chksum-((*(params.ref_D.data() + chk_start_idx)))));
      }

      // Atomic sum
      if(diff != 0){
        atomicAdd((SM_check_res + smid), diff);
      }
    }
    __syncthreads();

    // if(*(SM_check_res + smid)!=0){
    //   if(thread_idx == 0){
    //     // printf("%d,  Difference detected at SM %d. Reduced Sum: %d\n", iter, smid, *(SM_check_res+smid));
    //     // *(SM_check_res+smid) = 0;
    //   }
    // }
  }


  GemmBatched() = default;

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, 
                    int if_split_phase, int *SM_check_res, int partion) {

    // get SM id
    unsigned int real_smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
    int threadblock_tile_offset_m, threadblock_tile_offset_k, threadblock_tile_offset_n;

    // SM based schudule
    // assign enough SMs for each batch 
    int SM_per_batch = params.grid_tiled_shape.m() * params.grid_tiled_shape.n();
    if(SM_per_batch > 132){
      SM_per_batch = 132;
    }
    int batch_step = (int)(floor((double)132 / (double)SM_per_batch));
    int local_smid = real_smid % SM_per_batch;
    int init_batch_idx = real_smid / SM_per_batch;
    int batch_iter = (int)(ceil((double)params.batch_count / (double)batch_step));

    int smid = local_smid;

    // if(threadIdx.x == 0) printf("SM_per_batch: %d, batch_step: %d, batch_iter: %d\n", SM_per_batch, batch_step, batch_iter);

    // 2nd split SM for each matrix
    int checksumblk_per_col = 0;
    if(if_split_phase == 0 
      // || if_split_phase == 1
    ){
      // if able ABFT
      checksumblk_per_col = (int)(ceil((double)((partion) / (double)(128))));
    }
    int matrix_shape_m = params.grid_tiled_shape.m() - checksumblk_per_col;

    int max_col = (int)ceil((double)SM_per_batch / (double)(matrix_shape_m));
    if(max_col > params.grid_tiled_shape.n()){
      max_col = params.grid_tiled_shape.n();
    }

    int remaining_SM = (int)(max_col * checksumblk_per_col);
    int matrix_SM = (int)(SM_per_batch - remaining_SM);

    // int matrix_next_blk_offset_m = matrix_SM % (matrix_shape_m);
    // int matrix_next_blk_offset_n = (matrix_SM / matrix_shape_m);
    // int checksum_next_blk_offset_n = (checksumblk_per_col != 0) ? (remaining_SM / checksumblk_per_col) : 0;
    // // iteration based on GeMM not (GeMM + chksum)
    // int SM_iter = (int)ceil((double)((matrix_shape_m * params.grid_tiled_shape.n())/(double)matrix_SM));
    
    // if(threadIdx.x == 0) printf("matrix_SM: %d, M: %d, N: %d\n", matrix_SM, params.problem_size.m(), params.problem_size.n());

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
    
    threadblock_tile_offset_k = threadblock_tile_offset.k();

    // Early exit if CTA is out of range
    // if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
    //   params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

    //   return;
    // }

    int block_idx;
    int thread_idx = threadIdx.x;

    // check step
    // int check_step = 1;

    // Each CTA handles multiple batch indices to accommodate limited range of CUDA grid's Z dimension
    // for(int batch_idx = init_batch_idx; batch_idx < params.batch_count; batch_idx += batch_step) {
    for(int b_iter = 0; b_iter < batch_iter; b_iter += 1) {
      int batch_idx = init_batch_idx + b_iter * batch_step;
      // for(int iter = 0; iter < SM_iter; iter++){
      if(smid < matrix_SM){
        // for matrix
        int local_matrix_idx = smid;
        block_idx = local_matrix_idx + (local_matrix_idx / matrix_shape_m) * checksumblk_per_col;
        block_to_coordinate(block_idx, params.grid_tiled_shape.m(), threadblock_tile_offset_m, threadblock_tile_offset_n);
      }
      else{
        // for checksum          
        unsigned int local_chk_blk_idx = (smid - matrix_SM);
        block_to_coordinate(local_chk_blk_idx, checksumblk_per_col, threadblock_tile_offset_m, threadblock_tile_offset_n);
        threadblock_tile_offset_m += matrix_shape_m;
        block_idx = threadblock_tile_offset_m + threadblock_tile_offset_n * params.grid_tiled_shape.m();
      }
  
      // bool beyond_bound = false;  
      // // int iter = 0;

      // if(batch_idx >= params.batch_count || init_batch_idx >= batch_step){
      //   // return;
      //   beyond_bound = true;
      // }

      // Compute position within threadblock
      //  int thread_idx = threadIdx.x;
      //  int block_idx = threadblock_tile_offset_m + threadblock_tile_offset_n * params.grid_tiled_shape.m();
      
      if((batch_idx < params.batch_count) && (init_batch_idx < batch_step)){
        // Compute initial location in logical coordinates
        cutlass::MatrixCoord tb_offset_A{
          threadblock_tile_offset_m * Mma::Shape::kM,
          0
        };

        cutlass::MatrixCoord tb_offset_B{
          0,
          threadblock_tile_offset_n * Mma::Shape::kN
        };

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(
          params.params_A,
          params.ref_A.data(),
          params.problem_size.mk(),
          thread_idx,
          tb_offset_A);

        iterator_A.add_pointer_offset(params.stride_A * batch_idx);

        typename Mma::IteratorB iterator_B(
          params.params_B,
          params.ref_B.data(),
          params.problem_size.kn(),
          thread_idx,
          tb_offset_B);

        iterator_B.add_pointer_offset(params.stride_B * batch_idx);

        //
        // Main loop
        //

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = canonical_warp_idx_sync();

        int lane_idx = threadIdx.x % 32;
        
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;

        accumulators.clear();


        // Compute threadblock-scoped matrix multiply-add
        mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

        //
        // Epilogue
        //

        OutputOp output_op(params.epilogue);

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

        // if(thread_idx == 0){
        //   printf("sm id: %d, gridDim.z: %d, init_batch_idx: %d, batch_idx: %d, m_offset: %d, n_offset: %d\n", 
        //           smid, gridDim.z, threadblock_swizzle.get_batch_idx(), batch_idx, threadblock_offset.row(), threadblock_offset.column());
        // }

        // Tile iterator writing to output tile
        typename Epilogue::OutputTileIterator iterator_C(
          params.params_C,
          params.ref_C.data(),
          params.problem_size.mn(),
          thread_idx,
          threadblock_offset
        );

        iterator_C.add_pointer_offset(params.stride_C * batch_idx);

        // Tile iterator writing to output tile
        typename Epilogue::OutputTileIterator iterator_D(
          params.params_D,
          params.ref_D.data(),
          params.problem_size.mn(),
          thread_idx,
          threadblock_offset
        );

        iterator_D.add_pointer_offset(params.stride_D * batch_idx);

        Epilogue epilogue(
          shared_storage.epilogue, 
          thread_idx, 
          warp_idx, 
          lane_idx);

        // run efficient epilogue
        epilogue(output_op, iterator_D, accumulators, iterator_C);
      }

      // check phase
      #if 1
      if(if_split_phase == 0){
        if(b_iter == 0 && batch_iter != 1){
          continue;
        }
        else if(b_iter == (batch_iter - 1)){
          cooperative_groups::this_grid().sync();
        }
        
        if(init_batch_idx < batch_step){
          int *int_smem = reinterpret_cast<int *>(&shared_storage);
          int &next_matrix_block_idx = int_smem[0];
          int &next_chk_block_idx = int_smem[1];
          // int &flag = int_smem[2];
    
          int tmp_matrix_blk, tmp_chk_blk, tmp_flag;

          // block view
          if(thread_idx == 0){
            SM_based_schedule(params, threadblock_tile_offset_m, threadblock_tile_offset_n, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx, matrix_SM, 0, checksumblk_per_col);
    
            next_matrix_block_idx = tmp_matrix_blk;
            next_chk_block_idx = tmp_chk_blk;
            // flag = tmp_flag;
          }
          __syncthreads();
    
          // begin chkeck
          // if(flag == 1){
          if (smid < matrix_SM){
            int matrix_start_idx, chk_start_idx;
            // iter 1 ~ (n-2)
            if(b_iter < (batch_iter - 1) && b_iter > 0){
              int previous_batch_idx = batch_idx - batch_step;
              
              curr_iter_chk_offsets(params, matrix_start_idx, chk_start_idx, next_matrix_block_idx, next_chk_block_idx, 
                                      checksumblk_per_col, thread_idx, previous_batch_idx);

              check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, 
                          b_iter, real_smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx, batch_idx, threadblock_tile_offset_n);
            }
            // iter n-1
            else if(b_iter == batch_iter - 1){
              if(batch_iter != 1){
                //check last batch
                int previous_batch_idx = batch_idx - batch_step;
                curr_iter_chk_offsets(params, matrix_start_idx, chk_start_idx, next_matrix_block_idx, next_chk_block_idx, 
                                          checksumblk_per_col, thread_idx, previous_batch_idx);

                check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, 
                                      (b_iter-1), real_smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx, batch_idx, threadblock_tile_offset_n);
              }
              // cooperative_groups::this_grid().sync();
              if(batch_idx >= params.batch_count){
                return;
              }
                curr_iter_chk_offsets(params, matrix_start_idx, chk_start_idx, next_matrix_block_idx, next_chk_block_idx, 
                                          checksumblk_per_col, thread_idx, batch_idx);
                check_phase(params, matrix_start_idx, chk_start_idx, SM_check_res, 
                                    b_iter, real_smid, thread_idx, next_matrix_block_idx, next_chk_block_idx, block_idx, batch_idx, threadblock_tile_offset_n);
            }
          }
          // }
        }
      }
      else if(if_split_phase == 1){
        if((batch_idx < params.batch_count) && (init_batch_idx < batch_step)){
          checksumblk_per_col = (int)(ceil((double)((partion) / (double)(128))));

          int *int_smem = reinterpret_cast<int *>(&shared_storage);
          int &previous_matrix_block_idx = int_smem[0];
          int &previous_chk_block_idx = int_smem[1];

          // float *int_smem = reinterpret_cast<float *>(&shared_storage);
          // float &previous_matrix_block_idx = int_smem[0];
          // float &previous_chk_block_idx = int_smem[1];

          if(thread_idx == 0){
            int tmp_matrix_blk, tmp_chk_blk;
            
            previous_blks_idx(params, threadblock_tile_offset_m, tmp_matrix_blk, tmp_chk_blk, real_smid, block_idx, matrix_SM, b_iter, checksumblk_per_col);

            previous_matrix_block_idx = (float)tmp_matrix_blk;
            previous_chk_block_idx = (float)tmp_chk_blk;
          }
          __syncthreads();

          // int new_blk_idx = block_idx + threadblock_tile_offset_n * checksumblk_per_col;
          // if(thread_idx == 0) printf("smid: %d, blk: %d, newblk: %d, pre_blk: %d, pre_chk: %d\n", real_smid, block_idx, new_blk_idx, (int)(*int_smem), (int)(*(int_smem+1)));

          update_col(params, previous_matrix_block_idx, previous_chk_block_idx, threadblock_tile_offset_n, batch_idx, thread_idx, checksumblk_per_col);
          // update_col_v2(params, int_smem, threadblock_tile_offset_n, batch_idx, thread_idx, checksumblk_per_col);
        }
      }
      else{

      }
      #endif
      // }
    }
    // if(threadIdx.x == 0) printf("init_batch: %d\n", temp);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass