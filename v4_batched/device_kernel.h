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
// #include <cmath>
// #include "cutlass/gemm_ring_queue.h"

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

// __global__ void initQueues(RingQueue* queues, int** buffers, int cap) {
//   int idx = blockIdx.x;
//   if (idx < gridDim.x) {
//       // queues[idx].initial(buffers[idx], cap);
//   }
// }

/// Generic CUTLASS kernel template of Batched GEMM.
template <typename Operator>
CUTLASS_GLOBAL
void Kernel_Batched(typename Operator::Params params) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage);
  cutlass::arch::synclog_print();
}


/// Generic CUTLASS kernel template.
template <typename Operator>
CUTLASS_GLOBAL
void Kernel(typename Operator::Params params, 
            int if_split_phase, int *SM_check_res, int partion
            // int *all_start, int *compute, int *finding, int *recompute, int *compare, int *checking
          ) {  
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage, if_split_phase, SM_check_res, partion
    // all_start, compute, finding, recompute, compare, checking
  );
  
  cutlass::arch::synclog_print();
}

template <typename Operator>
CUTLASS_GLOBAL
void update_checksum(typename Operator::Params params){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM
  int matrix_SM = 128;
  int chk_SM = 132 - matrix_SM;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();

  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_SM));

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = (real_smid - matrix_SM) + b_iter * chk_SM;
    if(batch_idx < params.batch_count){
      // if(threadIdx.x == 0) {
      //   printf("smid: %d, batch idx: %d,\n", real_smid, batch_idx);
      // }

      int iter = (int)(ceil((double)N / (double)blockDim.y));
      for(int i = 0; i < iter; i++){
        int col_idx = (i * blockDim.y) + threadIdx.y;
        // int row_idx = threadIdx.x;
        if(col_idx < N){
          float accum = 0.f;
          int idx_a = (batch_idx * params.stride_A) + ((M + thread_idx) * K);
          int idx_b = (batch_idx * params.stride_B) + col_idx;
          int idx_chk = (batch_idx * params.stride_D) + (M + thread_idx) * N + col_idx;
        
          for(int k = 0; k < K; k++){
            float a = *(params.ref_A.data() + idx_a + k);
            float b = *(params.ref_B.data() + idx_b + k * N);
            accum += a * b;
          }
          *(params.ref_D.data() + idx_chk) = accum;
        }
      }
    }
  } 

  
  // for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
  //   int batch_idx = (real_smid - matrix_SM) + b_iter * chk_SM;
  //   if(batch_idx < params.batch_count){
  //     // if(threadIdx.x == 0) {
  //     //   printf("smid: %d, batch idx: %d,\n", real_smid, batch_idx);
  //     // }

  //     int col_idx = thread_idx;
  //     if(col_idx < N){
  //       for(int m = 0; m < 2; m++){
  //         float accum = 0.f;
  //         int idx_a = (batch_idx * params.stride_A) + ((M + m) * K);
  //         int idx_b = (batch_idx * params.stride_B) + col_idx;
  //         int idx_chk = (batch_idx * params.stride_D) + (M + m) * N + col_idx;
        
  //         for(int k = 0; k < K; k++){
  //           float a = *(params.ref_A.data() + idx_a + k);
  //           float b = *(params.ref_B.data() + idx_b + k * N);
  //           accum += a * b;
  //         }
  //         *(params.ref_D.data() + idx_chk) = accum;
  //       }
  //     }
  //   }
  // } 
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