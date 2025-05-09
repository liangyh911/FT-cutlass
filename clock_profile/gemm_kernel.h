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
  Gemm() { } 

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

  __device__ void group_find_SM(Params const &params, cutlass::gemm::GemmCoord threadblock_tile_offset,
    uint8_t *Signature_Array, int *Lock_Signature, int &tmp_matrix_blk, int &tmp_chk_blk, int &tmp_flag,
    unsigned int smid, int block_idx, int num_blk_per_group, int *SM_JOBS){
  if (threadblock_tile_offset.m() != (params.grid_tiled_shape.m() - 1)){
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
    int new_blk_idx = block_idx - threadblock_tile_offset.n();
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

__device__ void queue_find_SM(Params const &params, cutlass::gemm::GemmCoord threadblock_tile_offset,
                                uint8_t *Signature_Array, int *Lock_Signature, int &tmp_matrix_blk, int &tmp_chk_blk, int &tmp_flag,
                                unsigned int smid, int block_idx, int num_blk_per_group, RingQueue_v2 *d_queues, int *SM_JOBS){
  if (threadblock_tile_offset.m() != (params.grid_tiled_shape.m() - 1)){
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

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, 
                  uint8_t *Signature_Array, int *Lock_Signature, 
                  int *final_sum, int if_split_phase, RingQueue_v2 *d_queues, int *SM_JOBS,
                  int *all_start, int *compute, int *finding, int *recompute, int *compare, int *checking) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(), 
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

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
    
    // get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    // get num of SM
    // unsigned int nsm;
    // asm volatile("mov.u32  %0, %nsmid;" : "=r"(nsm));
    // printf("Num of SM: %d\n", nsm);
    
    // get thread id in each SM
    // unsigned int tid;
    // asm volatile("mov.u32 %0,%tid.x;" : "=r"(tid));
    // printf("SM id: %d, tid: %d, thread_idx;%d\n", smid, tid, thread_idx);
    
    // int off_A = tb_offset_A.row() + tb_offset_A.column()*params.problem_size.m();
    // int off_B = tb_offset_B.row() + tb_offset_B.column()*params.problem_size.n();

    // int off_A = tb_offset_A.column() + tb_offset_A.row()*params.problem_size.k();
    // int off_B = tb_offset_B.column() + tb_offset_B.row()*params.problem_size.n();
    
    // printf("M: %d, N: %d, K: %d \n", params.problem_size.m(), problem_size_k, params.problem_size.n());

    __syncthreads();
    if(thread_idx == 0 && (threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m()) == 0){
      *(all_start) = clock();
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
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    // printf("SM id: %d, A_data: %f, A_row_offset: %d, A_col_offset: %d, B_data: %f, B_row_offset: %d, B_col_offset: %d, C_row_offset: %d, C_col_offset: %d\n", 
    //   smid, *(params.ref_A.data()+off_A), tb_offset_A.row(), tb_offset_A.column(), 
    //         *(params.ref_B.data()+off_B), tb_offset_B.row(), tb_offset_B.column(),
    //         threadblock_offset.row(), threadblock_offset.column());

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // if(threadIdx.x == 0){
    //   printf("SM id: %d, block id: %d\n", smid, block_idx);
    // }

    // printf("block_idx: %d, tile_offset.m: %d, title_offset.n: %d, grid_tile_shape.m: %d, grid_tile_shape.n: %d\n", 
    //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), params.grid_tiled_shape.m(), params.grid_tiled_shape.n());

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
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
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

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

    // int off_C = threadblock_offset.row() + threadblock_offset.column() * params.problem_size.m();
    // int off_C = threadblock_offset.column() + threadblock_offset.row() * params.problem_size.n();

    // printf("SM id: %d, Block idx: %d, A_data: %f, A_row_offset: %d, A_col_offset: %d, B_data: %f, B_row_offset: %d, B_col_offset: %d, D_data: %f, D_row_offset: %d, D_col_offset: %d\n", 
    //           smid, block_idx, *(params.ref_A.data()+off_A), tb_offset_A.row(), tb_offset_A.column(), 
    //                           *(params.ref_B.data()+off_B), tb_offset_B.row(), tb_offset_B.column(),
    //                           *(params.ref_D.data()+off_C), threadblock_offset.row(), threadblock_offset.column());

    // 
    // Signature and Find (Matrix SM) (Column Checksum Only)
    //
    // __shared__ unsigned int next_chk_smid, next_matrix_smid;
    __syncthreads();
    if(thread_idx == 0 && block_idx == 0){
      *(compute) = clock();
    }
    __syncthreads();
    if(if_split_phase == 0){
      // __shared__ int next_matrix_block_idx, next_chk_block_idx, flag;
      int *int_smem = reinterpret_cast<int *>(&shared_storage);
      int &next_matrix_block_idx = int_smem[0];
      int &next_chk_block_idx = int_smem[1];
      int &flag = int_smem[2];

      int tmp_matrix_blk, tmp_chk_blk, tmp_flag;
      // block view
      if(thread_idx == 0){
        // RingQueue *queue = &d_queues[smid];
        // d_queues->enqueue(smid, smid);

        // int group_partition = 2;
        int group_partition =  (params.grid_tiled_shape.m() - 1) * params.grid_tiled_shape.n();
        // find_SM(params, threadblock_tile_offset,Signature_Array, Lock_Signature, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx);
        // group_find_SM(params, threadblock_tile_offset,Signature_Array, Lock_Signature, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx, group_partition, SM_JOBS);
        queue_find_SM(params, threadblock_tile_offset,Signature_Array, Lock_Signature, tmp_matrix_blk, tmp_chk_blk, tmp_flag, smid, block_idx, group_partition, d_queues, SM_JOBS);
        
        next_matrix_block_idx = tmp_matrix_blk;
        next_chk_block_idx = tmp_chk_blk;
        flag = tmp_flag;

        // int value; 
        // if(d_queues->dequeue(smid, &value)){
        //   printf("SM %d dequeued value: %d\n", smid, value);
        // }
      }
      __syncthreads();
      if(thread_idx == 0 && block_idx == 0){
        *(finding) = clock();
      }

      // begin chkeck
      if(flag == 1){
        if (threadblock_tile_offset.m() != (params.grid_tiled_shape.m() - 1)){
          int MatrixColBlkOffset = ((next_matrix_block_idx + 1) / params.grid_tiled_shape.m());
          int MatrixRowBlkOffset = ((next_matrix_block_idx + 1) % params.grid_tiled_shape.m() - 1);
          int matrix_start_idx = (MatrixColBlkOffset * 128) + (MatrixRowBlkOffset * 128) * params.problem_size.n() + thread_idx;

          int ChkColBlkOffset = ((next_chk_block_idx + 1) / params.grid_tiled_shape.m()) - 1;
          int ChkRowBlkOffset = (params.grid_tiled_shape.m() - 1);
          int chk_start_idx = (ChkColBlkOffset * 128) + (ChkRowBlkOffset * 128 + 2 * MatrixRowBlkOffset) * params.problem_size.n() + thread_idx;
          
          float recomputed_chksum = 0;
          int diff = 0;
          
          // if use group, not unroll
          #pragma unroll
          for(int r = 0; r < 128; r++){
            int idx = matrix_start_idx + r * params.problem_size.n();
            recomputed_chksum += *(params.ref_D.data() + idx);
          }
          __syncthreads();
          if(thread_idx == 0 && block_idx == 0){
            *(recompute) = clock();
          }
          
          if(fabs(recomputed_chksum - (*(params.ref_D.data() + chk_start_idx))) > (float)1e3){
            diff = 1;
            // printf("Difference detected at (%d, %d). matrix sum: (%d, %f), next chk: (%d, %f)\n", 
            //           smid, thread_idx, next_matrix_block_idx, recomputed_chksum, next_chk_block_idx, *(params.ref_D.data() + chk_start_idx));
          }
          __syncthreads();
          if(thread_idx == 0 && block_idx == 0){
            *(compare) = clock();
          }
          // Cooperative Groups Reduce
          // __shared__ int temp[128];
          int &temp = int_smem[3];
          auto g = this_thread_block();
          int block_sum = reduce_sum(g, &temp, diff);

          if(g.thread_rank() == 0){
            atomicAdd((final_sum + block_idx), block_sum);
            if(*(final_sum + block_idx) != 0){
              // printf("Difference detected at SM %d. Reduced Sum: %d\n", smid, *(final_sum + block_idx));
            }
            // else{
            //   printf("No difference detected at SM %d. Reduced Sum: %d\n", smid, *(final_sum + block_idx));
            // }
          }
        }
      }
      __syncthreads();
      if(thread_idx == 0 && block_idx == 0){
        *(checking) = clock();
        // printf("checking: %d\n", *checking);
      }
    }
    else if(if_split_phase == 1 && block_idx == 0){
      // 
      *(Signature_Array + block_idx) = (uint8_t)smid;
    }
    else{
      __syncthreads();
      if(thread_idx == 0 && block_idx == 0){
        *(checking) = clock();
      }
    }
    
    

    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass