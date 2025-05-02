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
    \brief Template for generic CUTLASS kernel.
*/

#pragma once

#include <cutlass/detail/helper_macros.hpp> // CUTLASS_HOST_DEVICE
#include <cutlass/platform/platform.h> // uint64_t
#include <cooperative_groups.h>
#include <cmath>
#include "cutlass/gemm_ring_queue.h"

using namespace cooperative_groups;

// __grid_constant__ was introduced in CUDA 11.7.
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7))) && !CUTLASS_CLANG_CUDA
#  define CUTLASS_GRID_CONSTANT_SUPPORTED
#endif

// __grid_constant__ can be enabled only on SM70+
#if defined(CUTLASS_GRID_CONSTANT_SUPPORTED) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#  define CUTLASS_GRID_CONSTANT_ENABLED
#endif

#if ! defined(CUTLASS_GRID_CONSTANT)
#  if defined(CUTLASS_GRID_CONSTANT_ENABLED)
#    define CUTLASS_GRID_CONSTANT __grid_constant__
#  else
#    define CUTLASS_GRID_CONSTANT
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

template <typename T>   struct Type2Type  {  using type=T;                    };
// using the simple type to replace the complex type to reduce this symbol size
template <typename  T>                                                                        struct GetUnderlyingKernel                              : public Type2Type<T>               {};
template <uint64_t shader_guid, unsigned index, template <uint64_t, unsigned> class Wrapper > struct GetUnderlyingKernel<Wrapper<shader_guid,index>>  : public Wrapper<shader_guid,index> {};
template <typename  T>                                                                        using  GetUnderlyingKernel_t                            = typename GetUnderlyingKernel<T>::type;


////////////////////////////////////////////////////////////////////////////////

__global__ void initQueues(RingQueue* queues, int** buffers, int cap) {
  int idx = blockIdx.x;
  if (idx < gridDim.x) {
      queues[idx].initial(buffers[idx], cap);
  }
}

/// Generic CUTLASS kernel template.
template <typename Operator>
CUTLASS_GLOBAL
void Kernel(typename Operator::Params params, uint8_t *Signature_Array, 
            int *Lock_Signature, int *final_sum, int if_split_phase, RingQueue *d_queues, int *SM_JOBS,
            int *all_start, int *compute, int *finding, int *checking) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage, Signature_Array, Lock_Signature, final_sum, if_split_phase, d_queues, SM_JOBS,
    all_start, compute, finding, checking);
  cutlass::arch::synclog_print();
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

// Check between SM
template <typename Operator>
CUTLASS_GLOBAL
void check_between_SM(typename Operator::Params params, uint8_t *Signature_Array, 
                        int *Lock_Signature, int *final_sum, int num_blk_per_group,
                        int *d_all_start_for_split, int *d_finding, int *d_checking, int *d_SM_JOBS){
  int thread_idx = threadIdx.x;
  int block_idx = blockIdx.x + gridDim.x * blockIdx.y;

  __shared__ int next_matrix_block_idx, next_chk_block_idx, flag;
  int tmp_matrix_blk, tmp_chk_blk, tmp_flag;
  unsigned int smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
  
  __syncthreads();
  if (thread_idx == 0){
    *(d_all_start_for_split+smid) = clock();
    // printf("block idx: %d, block idx x: %d \n", block_idx, blockIdx.x);
    if (blockIdx.x != (params.grid_tiled_shape.m() - 1)){
      *(d_SM_JOBS+smid) = 1;

      unsigned int next_matrix_smid, next_chk_smid;
      uint8_t matrix_block_idx = block_idx;
      uint8_t chk_block_idx;

      // int num_blk_per_group = 2;      
      int new_blk_idx = block_idx - blockIdx.y;
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

      // int next_local_blk_idx = local_blk_idx;
      // bool need_lock = true;
      // while (need_lock) {
      //   // matrix_block_idx = (matrix_block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
      //   next_local_blk_idx = (next_local_blk_idx + 1) % num_blk_per_group;
      //   int next_global_blk_idx = next_local_blk_idx + (group_idx * num_blk_per_group);
      //   matrix_block_idx = next_global_blk_idx + blockIdx.y;
        
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
        //         block_idx, blockIdx.x, blockIdx.y, smid, next_matrix_smid, tmp_matrix_blk, next_chk_smid, tmp_chk_blk);
      }
    }
    else{
      *(d_SM_JOBS+smid) = 2;
    }
    next_matrix_block_idx = tmp_matrix_blk;
    next_chk_block_idx = tmp_chk_blk;
    flag = tmp_flag;
  }
  __syncthreads();
  if(thread_idx == 0){
    *(d_finding + smid) = clock();
  }

  // begin chkeck
  if(flag == 1){
    int MatrixColBlkOffset = ((next_matrix_block_idx + 1) / params.grid_tiled_shape.m());
    int MatrixRowBlkOffset = ((next_matrix_block_idx + 1) % params.grid_tiled_shape.m() - 1);
    int matrix_start_idx = (MatrixColBlkOffset * 128) + (MatrixRowBlkOffset * 128) * params.problem_size.n() + thread_idx;

    int ChkColBlkOffset = ((next_chk_block_idx + 1) / params.grid_tiled_shape.m()) - 1;
    int ChkRowBlkOffset = (params.grid_tiled_shape.m() - 1);
    int chk_start_idx = (ChkColBlkOffset * 128) + (ChkRowBlkOffset * 128 + 2 * MatrixRowBlkOffset) * params.problem_size.n() + thread_idx;
    
    float recomputed_chksum = 0;
    int diff = 0;
    
    #pragma unroll
    for(int r = 0; r < 128; r++){
      int idx = matrix_start_idx + r * params.problem_size.n();
      recomputed_chksum += *(params.ref_D.data() + idx);
    }
    if(fabs(recomputed_chksum - (*(params.ref_D.data() + chk_start_idx))) > (float)1e3){
      diff = 1;
      printf("Difference detected at (%d, %d). matrix sum: (%d, %f), next chk: (%d, %f)\n", 
                smid, thread_idx, next_matrix_block_idx, recomputed_chksum, next_chk_block_idx, *(params.ref_D.data() + chk_start_idx));
    }
    // Cooperative Groups Reduce
    __shared__ int temp[128];
    auto g = this_thread_block();
    int block_sum = reduce_sum(g, temp, diff);

    if(g.thread_rank() == 0){
      atomicAdd((final_sum + block_idx), block_sum);
      if(*(final_sum + block_idx) != 0){
        printf("Difference detected at SM %d. Reduced Sum: %d\n", smid, *(final_sum + block_idx));
      }
      // else{
      //   printf("No difference detected at SM %d. Reduced Sum: %d\n", smid, *(final_sum + block_idx));
      // }
    }
  }
  __syncthreads();
  if(thread_idx == 0){
    *(d_checking + smid) = clock();
  }
}


/// Generic CUTLASS kernel template.
template <typename Operator>
CUTLASS_GLOBAL
void Kernel2(typename Operator::Params params) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator::invoke(params, *shared_storage);
  cutlass::arch::synclog_print();

}


////////////////////////////////////////////////////////////////////////////////
//
// 3.0 specific launch
//
////////////////////////////////////////////////////////////////////////////////

/// Generic CUTLASS kernel template.
template <typename Operator>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params)
{
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();

}

////////////////////////////////////////////////////////////////////////////////
} /// namespace cutlass
