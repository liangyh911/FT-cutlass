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

  __device__ int reduce_sum(thread_group g, int *temp, int val){
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane<i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
  }

  __device__ void find_SM(Params const &params, cutlass::gemm::GemmCoord threadblock_tile_offset,
      uint8_t *Signature_Array, int *Lock_Signature, int &tmp_matrix_blk, int &tmp_chk_blk, int &tmp_flag,
      unsigned int smid, int block_idx){
    if (threadblock_tile_offset.m() != (params.grid_tiled_shape.m() - 1)){
      // Signature for Matrxi SM
      *(Signature_Array + block_idx) = (uint8_t)smid;
      
      // Find Finished (Naive: get next)
      unsigned int next_matrix_smid, next_chk_smid;
      uint8_t matrix_block_idx = block_idx;
      uint8_t chk_block_idx;
      // printf("SM id:%d, SM array: %d, 1st next: %d\n", smid, *(Signature_Array + smid), next_matrix_smid);
      // __syncthreads();

      // 
      matrix_block_idx = (matrix_block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
      if ((matrix_block_idx + 1) % params.grid_tiled_shape.m() == 0){
        matrix_block_idx = (matrix_block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
      }
      int n = (matrix_block_idx + 1) / params.grid_tiled_shape.m();
      chk_block_idx = params.grid_tiled_shape.m() * (n + 1) - 1;
      while(true){
        if (*(Signature_Array + matrix_block_idx) != 255 && *(Signature_Array + chk_block_idx) != 255){
          next_matrix_smid = *(Signature_Array + matrix_block_idx);
          next_chk_smid = *(Signature_Array + chk_block_idx);

          tmp_matrix_blk = matrix_block_idx;
          tmp_chk_blk = chk_block_idx;
          break;
        }
      }

      // 
      // bool need_lock = true;
      // while (need_lock) {
      //   matrix_block_idx = (matrix_block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
      //   // lock for matrix SM selection
      //   if (atomicCAS((Lock_Signature + matrix_block_idx), 0, 1) == 0) {
      //     // get the corresponding chksum SM blk index
      //     int n = (matrix_block_idx + 1) / params.grid_tiled_shape.m();
      //     chk_block_idx = params.grid_tiled_shape.m() * (n + 1) - 1;
      //     // lock for the chksum SM
      //     // if (atomicCAS((Lock_Signature + chk_block_idx), 0, 1) == 0) {
      //       if ((matrix_block_idx + 1) % params.grid_tiled_shape.m() != 0 &&
      //           *(Signature_Array + matrix_block_idx) != 255 && 
      //           *(Signature_Array + chk_block_idx) != 255 &&
      //           *(Signature_Array + chk_block_idx) != smid) {
              
      //         next_matrix_smid = *(Signature_Array + matrix_block_idx);
      //         next_chk_smid = *(Signature_Array + chk_block_idx);

      //         tmp_matrix_blk = matrix_block_idx;
      //         tmp_chk_blk = chk_block_idx;

      //         *(Signature_Array + matrix_block_idx) = 255;
      //         need_lock = false;
      //       }
      //       // Release the lock
      //       // atomicExch((Lock_Signature + chk_block_idx), 0);
      //       // printf("current SM: %d, next SM: %d\n", smid, next_matrix_smid);
      //     // }
      //     atomicExch((Lock_Signature + matrix_block_idx), 0);
      //   }
      // }

      // Check chksum smid == matrix smid
      if(next_chk_smid == next_matrix_smid){
        tmp_flag = 0;
        printf("Recompute chksum using current SM\n");
      }
      // SM ids are not the same
      else{
        tmp_flag = 1;
        // printf("Check\n");
        // printf("Check. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix SM: (%d, %d), next chk SM: (%d, %d)\n", 
        //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), smid, next_matrix_smid, tmp_matrix_blk, next_chk_smid, tmp_chk_blk);
      }
    }
    else{
      // Signature for Checksum (encoded) SM
      *(Signature_Array + block_idx) = (uint8_t)smid;
      // printf("chksum. block_idx: %d, tile_offset.m: %d, title_offset.n: %d, SM: %d, \n", 
      //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), *(Signature_Array + block_idx));
    }
  }

  __device__ void group_find_SM(Params const &params, int threadblock_tile_offset_m, int threadblock_tile_offset_n,
    uint8_t *Signature_Array, int *Lock_Signature, int &tmp_matrix_blk, int &tmp_chk_blk, int &tmp_flag,
    unsigned int smid, int block_idx, int num_blk_per_group, int *SM_JOBS){
  if (threadblock_tile_offset_m != (params.grid_tiled_shape.m() - 1)){
    // Signature for Matrxi SM
    // *(SM_JOBS + smid) = 1;
    *(Signature_Array + block_idx) = (uint8_t)smid;
    
    // Find Finished (Naive: get next)
    unsigned int next_matrix_smid, next_chk_smid;
    uint8_t matrix_block_idx, chk_block_idx;
    // printf("SM id:%d, SM array: %d, 1st next: %d\n", smid, *(Signature_Array + smid), next_matrix_smid);
    // __syncthreads();

    // 
    // int num_blk_per_group = 2;
    int new_blk_idx = block_idx - threadblock_tile_offset_n;
    int group_idx = new_blk_idx / num_blk_per_group;

    int num_group = (params.grid_tiled_shape.m() - 1) * params.grid_tiled_shape.n() / num_blk_per_group;
    int remaining_blk = (params.grid_tiled_shape.m() - 1) * params.grid_tiled_shape.n() % num_blk_per_group;
    int previous_blk_size = num_blk_per_group;
    if(remaining_blk == 1){
      if(group_idx == (num_group-1)){
        num_blk_per_group++;
      }
      if(group_idx == num_group){
        group_idx--;
        num_blk_per_group++;
      }
    }
    else if(remaining_blk > 1){
      if(group_idx == num_group){
        num_blk_per_group = remaining_blk;
      }
    }

    int local_blk_idx = new_blk_idx % previous_blk_size;
    int next_local_blk_idx = (local_blk_idx + 1) % num_blk_per_group;
    int next_global_blk_idx = next_local_blk_idx + (group_idx * previous_blk_size);
    int new_offset_n = next_global_blk_idx / (params.grid_tiled_shape.m() - 1);
    matrix_block_idx = next_global_blk_idx + new_offset_n;
    // 
    // matrix_block_idx = (matrix_block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
    if ((matrix_block_idx + 1) % params.grid_tiled_shape.m() == 0){
      matrix_block_idx = (matrix_block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
    }
    int n = (matrix_block_idx + 1) / params.grid_tiled_shape.m();
    chk_block_idx = params.grid_tiled_shape.m() * (n + 1) - 1;
    while(true){
      if (*(Signature_Array + matrix_block_idx) != 255 && *(Signature_Array + chk_block_idx) != 255){
        next_matrix_smid = *(Signature_Array + matrix_block_idx);
        next_chk_smid = *(Signature_Array + chk_block_idx);

        tmp_matrix_blk = matrix_block_idx;
        tmp_chk_blk = chk_block_idx;
        break;
      }
    }

    // 
    // int next_local_blk_idx = local_blk_idx;
    // bool need_lock = true;
    // while (need_lock) {
    //   // matrix_block_idx = (matrix_block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
    //   next_local_blk_idx = (next_local_blk_idx + 1) % num_blk_per_group;
    //   int next_global_blk_idx = next_local_blk_idx + (group_idx * num_blk_per_group);
    //   matrix_block_idx = next_global_blk_idx + threadblock_tile_offset.n();
    //   //
    //   // lock for matrix SM selection
    //   if (atomicCAS((Lock_Signature + matrix_block_idx), 0, 1) == 0) {
    //     // get the corresponding chksum SM blk index
    //     int n = (matrix_block_idx + 1) / params.grid_tiled_shape.m();
    //     chk_block_idx = params.grid_tiled_shape.m() * (n + 1) - 1;
    //     // lock for the chksum SM
    //     // if (atomicCAS((Lock_Signature + chk_block_idx), 0, 1) == 0) {
    //       if ((matrix_block_idx + 1) % params.grid_tiled_shape.m() != 0 &&
    //           *(Signature_Array + matrix_block_idx) != 255 && 
    //           *(Signature_Array + chk_block_idx) != 255 &&
    //           *(Signature_Array + chk_block_idx) != smid) {
            
    //         next_matrix_smid = *(Signature_Array + matrix_block_idx);
    //         next_chk_smid = *(Signature_Array + chk_block_idx);

    //         tmp_matrix_blk = matrix_block_idx;
    //         tmp_chk_blk = chk_block_idx;

    //         *(Signature_Array + matrix_block_idx) = 255;
    //         need_lock = false;
    //       }
    //       // Release the lock
    //       // atomicExch((Lock_Signature + chk_block_idx), 0);
    //       // printf("current SM: %d, next SM: %d\n", smid, next_matrix_smid);
    //     // }
    //     atomicExch((Lock_Signature + matrix_block_idx), 0);
    //   }
    // }

    // Check chksum smid == matrix smid
    if(next_chk_smid == next_matrix_smid){
      tmp_flag = 0;
      printf("Recompute chksum using current SM\n");
      // printf("---Recompute chksum using current SM. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix SM: (%d, %d), next chk SM: (%d, %d)\n", 
      //   block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), smid, next_matrix_smid, tmp_matrix_blk, next_chk_smid, tmp_chk_blk);

    }
    // SM ids are not the same
    else{
      tmp_flag = 1;
      // printf("Check\n");
      // printf("Check. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix SM: (%d, %d), next chk SM: (%d, %d)\n", 
      //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), smid, next_matrix_smid, tmp_matrix_blk, next_chk_smid, tmp_chk_blk);
    }
  }
  else{
    // Signature for Checksum (encoded) SM
    // *(SM_JOBS + smid) = 2;
    *(Signature_Array + block_idx) = (uint8_t)smid;
    // printf("chksum. block_idx: %d, tile_offset.m: %d, title_offset.n: %d, SM: %d, \n", 
    //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), *(Signature_Array + block_idx));
  }
}

__device__ int unroll(Params const &params, int matrix_start_idx){
  int recomputed_chksum = 0;

  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 0 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 1 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 2 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 3 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 4 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 5 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 6 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 7 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 8 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 9 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 10 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 11 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 12 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 13 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 14 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 15 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 16 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 17 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 18 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 19 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 20 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 21 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 22 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 23 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 24 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 25 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 26 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 27 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 28 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 29 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 30 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 31 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 32 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 33 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 34 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 35 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 36 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 37 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 38 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 39 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 40 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 41 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 42 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 43 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 44 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 45 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 46 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 47 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 48 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 49 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 50 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 51 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 52 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 53 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 54 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 55 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 56 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 57 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 58 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 59 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 60 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 61 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 62 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 63 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 64 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 65 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 66 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 67 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 68 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 69 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 70 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 71 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 72 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 73 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 74 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 75 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 76 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 77 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 78 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 79 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 80 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 81 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 82 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 83 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 84 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 85 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 86 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 87 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 88 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 89 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 90 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 91 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 92 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 93 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 94 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 95 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 96 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 97 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 98 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 99 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 100 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 101 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 102 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 103 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 104 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 105 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 106 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 107 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 108 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 109 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 110 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 111 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 112 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 113 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 114 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 115 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 116 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 117 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 118 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 119 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 120 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 121 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 122 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 123 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 124 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 125 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 126 * params.problem_size.n());
  recomputed_chksum += *(params.ref_D.data() + matrix_start_idx + 127 * params.problem_size.n());
  
  return recomputed_chksum;
}

__device__ float unroll_small_loop(Params const &params, int matrix_start_idx, int n){
  float t1 = 0;
  #pragma unroll 
  for(int i = 0; i < 4; i++){
    t1 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t2 = 0;
  #pragma unroll 
  for(int i = 4; i < 8; i++){
    t2 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t3 = 0;
  #pragma unroll 
  for(int i = 8; i < 12; i++){
    t3 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t4 = 0;
  #pragma unroll 
  for(int i = 12; i < 16; i++){
    t4 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t5 = 0;
  #pragma unroll 
  for(int i = 16; i < 20; i++){
    t5 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t6 = 0;
  #pragma unroll 
  for(int i = 20; i < 24; i++){
    t6 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t7 = 0;
  #pragma unroll 
  for(int i = 24; i < 28; i++){
    t7 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t8 = 0;
  #pragma unroll 
  for(int i = 28; i < 32; i++){
    t8 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t9 = 0;
  #pragma unroll 1
  for(int i = 32; i < 36; i++){
    t9 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t10 = 0;
  #pragma unroll 
  for(int i = 36; i < 40; i++){
    t10 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t11 = 0;
  #pragma unroll 
  for(int i = 40; i < 44; i++){
    t11 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t12 = 0;
  #pragma unroll 
  for(int i = 44; i < 48; i++){
    t12 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t13 = 0;
  #pragma unroll 
  for(int i = 48; i < 52; i++){
    t13 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t14 = 0;
  #pragma unroll
  for(int i = 52; i < 56; i++){
    t14 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t15 = 0;
  #pragma unroll 1
  for(int i = 56; i < 60; i++){
    t15 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t16 = 0;
  #pragma unroll 
  for(int i = 60; i < 64; i++){
    t16 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t17 = 0;
  #pragma unroll 
  for(int i = 64; i < 68; i++){
    t17 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t18 = 0;
  #pragma unroll  
  for(int i = 68; i < 72; i++){
    t18 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t19 = 0;
  #pragma unroll 
  for(int i = 72; i < 76; i++){
    t19 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t20 = 0;
  #pragma unroll 
  for(int i = 76; i < 80; i++){
    t20 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t21 = 0;
  #pragma unroll 
  for(int i = 80; i < 84; i++){
    t21 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t22 = 0;
  #pragma unroll 
  for(int i = 84; i < 88; i++){
    t22 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t23 = 0;
  #pragma unroll 1
  for(int i = 88; i < 92; i++){
    t23 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t24 = 0;
  #pragma unroll 
  for(int i = 92; i < 96; i++){
    t24 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t25 = 0;
  #pragma unroll 
  for(int i = 96; i < 100; i++){
    t25 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t26 = 0;
  #pragma unroll 
  for(int i = 100; i < 104; i++){
    t26 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t27 = 0;
  #pragma unroll 
  for(int i = 104; i < 108; i++){
    t27 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t28 = 0;
  #pragma unroll 
  for(int i = 108; i < 112; i++){
    t28 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t29 = 0;
  #pragma unroll 
  for(int i = 112; i < 116; i++){
    t29 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t30 = 0;
  #pragma unroll 
  for(int i = 116; i < 120; i++){
    t30 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t31 = 0;
  #pragma unroll 
  for(int i = 120; i < 124; i++){
    t31 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  float t32 = 0;
  #pragma unroll 1
  for(int i = 124; i < 128; i++){
    t32 += *(params.ref_D.data() + matrix_start_idx + i * n);
  }

  return (t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12+t13+t14+t15+t16+
          t17+t18+t19+t20+t21+t22+t23+t24+t25+t26+t27+t28+t29+t30+t31+t32);
}

__device__ void queue_find_SM(Params const &params, int threadblock_tile_offset_m, int threadblock_tile_offset_n,
                                uint8_t *Signature_Array, int *Lock_Signature, int &tmp_matrix_blk, int &tmp_chk_blk, int &tmp_flag,
                                unsigned int smid, int block_idx, int num_blk_per_group, RingQueue_v2 *d_queues, int *SM_JOBS){
  if (threadblock_tile_offset_m != (params.grid_tiled_shape.m() - 1)){
    // *(SM_JOBS + smid) = 1;
    // *(Signature_Array + block_idx) = (uint8_t)smid;
    
    // Wait self-check finish and enqueue
    int n = (block_idx) / params.grid_tiled_shape.m();
    int chk_block_idx = params.grid_tiled_shape.m() * (n + 1) - 1;
    unsigned int next_chk_smid, next_matrix_smid;
    
    while(true){
      if(*(Signature_Array + chk_block_idx) != 255){
        d_queues->enqueue(smid, block_idx);
        break;
      }
    }

    // Select from next SM's queue
    // int count = 0;
    next_matrix_smid = (smid + 1) % 132;
    // RingQueue *next_queue;
    while(true){
      // d_queues->dequeue(next_matrix_smid, &tmp_matrix_blk);
      // break;
      if (d_queues->dequeue(next_matrix_smid, &tmp_matrix_blk)){
        int tmp = (tmp_matrix_blk) / params.grid_tiled_shape.m();
        tmp_chk_blk = params.grid_tiled_shape.m() * (tmp + 1) - 1;
        next_chk_smid = *(Signature_Array + tmp_chk_blk);
        
        if(tmp_matrix_blk == tmp_chk_blk){
          next_matrix_smid = (next_matrix_smid + 1) % 132;
          continue;
        }
        break;
      }
      // count++;
      // printf("cur SM:%d, next SM: %d, count: %d\n",smid, next_matrix_smid, count);
      // printf("%d\n", next_queue->tail);
    }
    // printf("%d, %d, %p\n", smid, tmp_matrix_blk, &tmp_matrix_blk);

    // next_matrix_smid = smid;
    // while(true){
    //   next_matrix_smid = (next_matrix_smid + 1) % 132;
    //   if(*(SM_JOBS + next_matrix_smid) == 1){
    //     RingQueue *next_queue = &d_queues[next_matrix_smid];
    //     if(next_queue->dequeue(&tmp_matrix_blk)){
    //       n = (tmp_matrix_blk) / params.grid_tiled_shape.m();
    //       tmp_chk_blk = params.grid_tiled_shape.m() * (n + 1) - 1;
    //       next_chk_smid = *(Signature_Array + tmp_chk_blk);
    //       if(tmp_matrix_blk == tmp_chk_blk){
    //         continue;
    //       }
    //       break;
    //     }
    //   }
    // }

    // if(!F){
    //   printf("------Empty queue. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix SM: (%d, %d), next chk SM: (%d, %d)\n", 
    //     block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), smid, next_matrix_smid, tmp_matrix_blk, next_chk_smid, tmp_chk_blk);
    // }
    
    // Check chksum smid == matrix smid
    if(next_chk_smid == next_matrix_smid){
      tmp_flag = 0;
      printf("Recompute chksum using current SM\n");
      // printf("---Recompute chksum using current SM. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix SM: (%d, %d), next chk SM: (%d, %d)\n", 
      //   block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), smid, next_matrix_smid, tmp_matrix_blk, next_chk_smid, tmp_chk_blk);
    }
    // SM ids are not the same
    else{
      tmp_flag = 1;
      // printf("Check\n");
      // printf("Check. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix SM: (%d, %d), next chk SM: (%d, %d)\n", 
      //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), smid, next_matrix_smid, tmp_matrix_blk, next_chk_smid, tmp_chk_blk);
    }
  }
  else{
    // Signature for Checksum (encoded) SM
    // *(SM_JOBS + smid) = 2;
    *(Signature_Array + block_idx) = (uint8_t)smid;
    d_queues->enqueue(smid, block_idx);

    // printf("chksum. block_idx: %d, tile_offset.m: %d, title_offset.n: %d, SM: %d, \n", 
    //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), *(Signature_Array + block_idx));
  }
}

__device__ void SM_based_schedule(Params const &params, int threadblock_tile_offset_m, int threadblock_tile_offset_n,
                                  int &tmp_matrix_blk, int &tmp_chk_blk, int &tmp_flag,
                                  unsigned int smid, int block_idx, int matrix_SM){
  if (threadblock_tile_offset_m != (params.grid_tiled_shape.m() - 1)){
    // issue there
    // tmp_matrix_blk = (block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
    
    int new_blk_idx = block_idx - threadblock_tile_offset_n;
    int group_idx = new_blk_idx / matrix_SM;

    int num_group = (params.grid_tiled_shape.m() - 1) * params.grid_tiled_shape.n() / matrix_SM;
    int remaining_blk = (params.grid_tiled_shape.m() - 1) * params.grid_tiled_shape.n() % matrix_SM;
    int previous_blk_size = matrix_SM;
    if(remaining_blk == 1){
      if(group_idx == (num_group-1)){
        matrix_SM++;
      }
      if(group_idx == num_group){
        group_idx--;
        matrix_SM++;
      }
    }
    else if(remaining_blk > 1){
      if(group_idx == num_group){
        matrix_SM = remaining_blk;
      }
    }

    int local_blk_idx = new_blk_idx % previous_blk_size;
    int next_local_blk_idx = (local_blk_idx + 1) % matrix_SM;
    int next_global_blk_idx = next_local_blk_idx + (group_idx * previous_blk_size);
    int new_offset_n = next_global_blk_idx / (params.grid_tiled_shape.m() - 1);
    tmp_matrix_blk = next_global_blk_idx + new_offset_n;
    
    if ((tmp_matrix_blk + 1) % params.grid_tiled_shape.m() == 0){
      tmp_matrix_blk = (tmp_matrix_blk + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
    }
    int n = (tmp_matrix_blk) / params.grid_tiled_shape.m();
    tmp_chk_blk = params.grid_tiled_shape.m() * (n + 1) - 1;
    tmp_flag = 1;

    printf("Check. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix block: (%d), next chk block: (%d)\n", 
            block_idx, threadblock_tile_offset_m, threadblock_tile_offset_n, smid, tmp_matrix_blk, tmp_chk_blk);

  }
  else{
  // printf("chksum. block_idx: %d, tile_offset.m: %d, title_offset.n: %d, SM: %d, \n", 
  //     block_idx, threadblock_tile_offset_m, threadblock_tile_offset_n, smid);
  }
}

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, 
                  uint8_t *Signature_Array, int *Lock_Signature, 
                  int *final_sum, int if_split_phase, RingQueue_v2 *d_queues, int *SM_JOBS, int *SM_schedule, int *SM_check_res,
                  int *all_start, int *compute, int *finding, int *recompute, int *compare, int *checking) {
    
    // get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    int threadblock_tile_offset_m, threadblock_tile_offset_k, threadblock_tile_offset_n;

    // if(threadIdx.x==0){
    //   printf("operator smid: %d\n", smid);
    // }
              
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    // if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
    //   params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

    //   return;
    // }

    // 
    // new offset based on SM id
    // 

    // if(threadIdx.x==0){
    //   printf("exit smid: %d\n", smid);
    // }
    
    // new offset - navie
    // threadblock_tile_offset_m = smid % gridDim.x;
    // threadblock_tile_offset_k = threadblock_tile_offset.k();
    // threadblock_tile_offset_n = smid / gridDim.x;

    for(int iter = 0; iter < *(SM_schedule+6); iter++){

    bool beyond_bound = false;
    // new offset - first Z matrix, last Y checksum
    int add_col = 0;
    // SM_schedule (num of SM for matrix, num of SM of chk, chk blk row, matrix offset_m, matrix offset_n, chk offset_n)
    if(smid < *SM_schedule){
      // for matrix
      threadblock_tile_offset_m = (smid % (params.grid_tiled_shape.m() - (*(SM_schedule+2))) + iter * (*(SM_schedule+3))) % (params.grid_tiled_shape.m() - (*(SM_schedule+2)));
      add_col = ((smid % (params.grid_tiled_shape.m() - (*(SM_schedule+2))) + iter * (*(SM_schedule+3))) / (params.grid_tiled_shape.m() - (*(SM_schedule+2))));
      threadblock_tile_offset_k = threadblock_tile_offset.k();
      threadblock_tile_offset_n = smid / (params.grid_tiled_shape.m() - (*(SM_schedule+2))) + iter * (*(SM_schedule+4)) + add_col;
    }
    else{
      // for checksum
      unsigned int local_chk_blk_idx = smid - *SM_schedule;
      threadblock_tile_offset_m = (params.grid_tiled_shape.m() - (*(SM_schedule+2))) + (local_chk_blk_idx % (*(SM_schedule+2)));
      threadblock_tile_offset_k = threadblock_tile_offset.k();
      threadblock_tile_offset_n = local_chk_blk_idx / (*(SM_schedule+2)) + iter * (*(SM_schedule+5));
    }
    if(threadblock_tile_offset_n >= params.grid_tiled_shape.n()){
      // return;
      beyond_bound = true;
    }
    
    // if(smid == 23 && threadIdx.x==0){
    //   printf("smid: %d, m: %d, n:%d\n", smid, threadblock_tile_offset_m,threadblock_tile_offset_n);
    // }

    // threadblock_tile_offset_m = threadblock_tile_offset.m();
    // threadblock_tile_offset_k = threadblock_tile_offset.k();
    // threadblock_tile_offset_n = threadblock_tile_offset.n();
    
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

    __syncthreads();
    if(thread_idx == 0 && smid == 0){
      *(all_start + iter) = clock();
      // printf("all_start: %d\n", *all_start);
    }

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
    
    // if(block_idx == 1 && thread_idx == 0){
    //   printf("accumulators:\n");
    //   for(int r = 0; r < 1; r++){
    //     for(int c = 0; c < 128; c++){
    //       int idx = 0 + (r + c * 128);
    //       printf("%f, ", *(accumulators.data() + idx));
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
    // if(block_idx == 0){
    //   if(thread_idx == 0){
    //     *(params.ref_D.data()+0) = 0;
    //   }
    //   __syncthreads();
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
    __syncthreads();
    if(thread_idx == 0 && smid == 0){
      *(compute + iter) = clock();
    }
    }
    // if(block_idx == 94){
    //   printf("iter: %d, thread: %d, value: %f\n", iter, thread_idx, *(params.ref_D.data() + thread_idx));
    // }
    if(iter == (*((SM_schedule)+6))-1){
      cooperative_groups::this_grid().sync();
    }
    // __syncthreads();

    if(beyond_bound){
      return;
    }

    if(if_split_phase == 0){
      // __shared__ int next_matrix_block_idx, next_chk_block_idx, flag;
      int *int_smem = reinterpret_cast<int *>(&shared_storage);
      int &next_matrix_block_idx = int_smem[0];
      int &next_chk_block_idx = int_smem[1];
      int &flag = int_smem[2];

      int tmp_matrix_blk, tmp_chk_blk, tmp_flag;
      // block view
      if(thread_idx == 0){
        // d_queues->enqueue(smid, smid);

        // int group_partition = (params.grid_tiled_shape.m() - 1);
        int group_partition =  (params.grid_tiled_shape.m() - 1) * params.grid_tiled_shape.n();
        // find_SM(params, threadblock_tile_offset,Signature_Array, Lock_Signature, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx);
        // group_find_SM(params, threadblock_tile_offset_m, threadblock_tile_offset_n, Signature_Array, Lock_Signature, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx, group_partition, SM_JOBS);
        // queue_find_SM(params, threadblock_tile_offset_m, threadblock_tile_offset_n, Signature_Array, Lock_Signature, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx, group_partition, d_queues, SM_JOBS);
        SM_based_schedule(params, threadblock_tile_offset_m, threadblock_tile_offset_n, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx, *(SM_schedule));

        next_matrix_block_idx = tmp_matrix_blk;
        next_chk_block_idx = tmp_chk_blk;
        flag = tmp_flag;

        // int value; 
        // if(d_queues->dequeue(smid, &value)){
        //   printf("SM %d dequeued value: %d\n", smid, value);
        // }
      }
      __syncthreads();
      if(thread_idx == 0 && smid == 0){
        *(finding + iter) = clock();
      }
      
      // int t = 0;
      // for(int i = 0; i < 10000; i++){
      //   int a = i*10;
      //   t = a;
      //   if(thread_idx == 0 && t == 100){
      //     printf("abcd\n");
      //   }
      // }

      // begin chkeck
      if(flag == 1){
        if (threadblock_tile_offset_m != (params.grid_tiled_shape.m() - 1)){
          int MatrixColBlkOffset, MatrixRowBlkOffset, matrix_start_idx, ChkColBlkOffset, ChkRowBlkOffset, chk_start_idx;
          
          if(iter < (*((SM_schedule)+6)) && iter > 0){
            MatrixColBlkOffset = next_matrix_block_idx / params.grid_tiled_shape.m() - (*(SM_schedule+4)) - add_col;
            MatrixRowBlkOffset = next_matrix_block_idx % params.grid_tiled_shape.m() - (*(SM_schedule+3));
            if(MatrixRowBlkOffset < 0){
              MatrixRowBlkOffset += (params.grid_tiled_shape.m() - (*(SM_schedule+2)));
            }
            matrix_start_idx = (MatrixColBlkOffset * 128) + (MatrixRowBlkOffset * 128) * params.problem_size.n() + thread_idx;

            ChkColBlkOffset = next_chk_block_idx / params.grid_tiled_shape.m() - (*(SM_schedule+4)) - add_col;
            ChkRowBlkOffset = (params.grid_tiled_shape.m() - (*(SM_schedule+2)));
            chk_start_idx = (ChkColBlkOffset * 128) + (ChkRowBlkOffset * 128 + 2 * MatrixRowBlkOffset) * params.problem_size.n() + thread_idx;
          }
          else if(iter == (*((SM_schedule)+6))-1){
            MatrixColBlkOffset = next_matrix_block_idx / params.grid_tiled_shape.m();
            MatrixRowBlkOffset = next_matrix_block_idx % params.grid_tiled_shape.m();
            matrix_start_idx = (MatrixColBlkOffset * 128) + (MatrixRowBlkOffset * 128) * params.problem_size.n() + thread_idx;

            ChkColBlkOffset = next_chk_block_idx / params.grid_tiled_shape.m();
            ChkRowBlkOffset = (params.grid_tiled_shape.m() - (*(SM_schedule+2)));
            chk_start_idx = (ChkColBlkOffset * 128) + (ChkRowBlkOffset * 128 + 2 * MatrixRowBlkOffset) * params.problem_size.n() + thread_idx;
          }
          else{
            continue;
          }

          float recomputed_chksum = 0;
          int diff = 0;

          // if(block_idx == 1 && thread_idx == 0){
          //   printf("%d, %d, %d\n", MatrixRowBlkOffset, MatrixColBlkOffset, matrix_start_idx);
          //   for(int r = 0; r < 1; r++){
          //     for(int c = 0; c < 128; c++){
          //       int idx = 256*128 + (c + r * params.problem_size.n());
          //       printf("%f, ", *(params.ref_D.data() + idx));
          //     }
          //     printf("\n");
          //   }
          // }
          
          // if use group, not unroll
          int N = params.problem_size.n();
          // void *p = params.ref_D.data();
          #pragma unroll
          for(int r = 0; r < 128; r++){
            int idx = matrix_start_idx + r * N;
            recomputed_chksum += *(params.ref_D.data() + idx);
            // float temp = params.ref_D.data(idx);
          }
          
          __syncthreads();
          if(thread_idx == 0 && smid == 0){
            *(recompute + iter) = clock();
          }
          
          if(fabs(recomputed_chksum - (*(params.ref_D.data() + chk_start_idx))) > (float)1e3){
            diff = 1;
            printf("%d Difference detected at (%d, %d). matrix sum: (%d, %f), next chk: (%d, %f)\n", 
                      iter, smid, thread_idx, next_matrix_block_idx, recomputed_chksum, next_chk_block_idx, *(params.ref_D.data() + chk_start_idx));
          }
          __syncthreads();
          if(thread_idx == 0 && smid == 0){
            *(compare + iter) = clock();
          }

          // Cooperative Groups Reduce
          // int &temp = int_smem[3];
          // auto g = this_thread_block();
          // int block_sum = reduce_sum(g, &temp, diff);
          // if(g.thread_rank() == 0){
          //   atomicAdd((final_sum + block_idx), block_sum);
          //   if(*(final_sum + block_idx) != 0){
          //     printf("Difference detected at iteration: %d, at SM %d. Reduced Sum: %d\n", iter, smid, *(final_sum + block_idx));
          //   }
          //   // else{
          //   //   printf("No difference detected at SM %d. Reduced Sum: %d\n", smid, *(final_sum + block_idx));
          //   // }
          // }

          // Atomic sum
          if(diff != 0){
            atomicAdd((SM_check_res+smid), diff);
          }
          __syncthreads();
          if(*(SM_check_res+smid)!=0){
            if(thread_idx == 0){
              // printf("Difference detected at SM %d. Reduced Sum: %d\n", smid, *(SM_check_res+smid));
            }
          }

          __syncthreads();
          if(thread_idx == 0 && smid == 0){
            *(checking + iter) = clock();
            // printf("checking: %d\n", *(checking + iter));
          }
        }
      }
    }
    else if(if_split_phase == 1 && block_idx == 0){
      // 
      *(Signature_Array + block_idx) = (uint8_t)smid;
    }
    else{
      __syncthreads();
      if(thread_idx == 0 && smid == 0){
        *(checking + iter) = clock();
      }
    }

    // __syncthreads();
    // if(thread_idx == 0){
    //   threadblock_buffer[0] = threadblock_tile_offset_m;
    //   threadblock_buffer[1] = threadblock_tile_offset_k;
    //   threadblock_buffer[2] = threadblock_tile_offset_n;
    // }
    // }
    
    
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