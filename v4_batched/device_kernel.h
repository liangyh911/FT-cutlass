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
            int if_split_phase, int *SM_check_res, int partion, int matrix_SM
            // int *all_start, int *compute, int *finding, int *recompute, int *compare, int *checking
          ) {  
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage, if_split_phase, SM_check_res, partion, matrix_SM
    // all_start, compute, finding, recompute, compare, checking
  );
  
  cutlass::arch::synclog_print();
}

template <typename Operator>
CUTLASS_GLOBAL
void update_checksum(typename Operator::Params params, int matrix_SM){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;
  int chk_SM = 132 - matrix_SM;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  extern __shared__ float SharedMem[];
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();

  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_SM));

  int local_smid = (real_smid - matrix_SM);
  int col_idx = threadIdx.x;

  int mk = M * K;
  int mn = M * N;
  int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;

  int load_iter = (int)(ceil((double)(2 * K)/ (double)blockDim.x));

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_SM;
    if(batch_idx < params.batch_count){
      // if(threadIdx.x == 0) {
      //   printf("smid: %d, batch idx: %d,\n", real_smid, batch_idx);
      // }
    
      float accum1 = 0.f;
      float accum2 = 0.f;
      
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      // int idx_a_2 = (batch_idx * params.stride_A) + m1k;

      int idx_b = (batch_idx * params.stride_B) + col_idx;

      int idx_chk_1 = (batch_idx * params.stride_D + mn) + col_idx;
      int idx_chk_2 = (batch_idx * params.stride_D) + m1n + col_idx;

      // load checksum to share memroy
      __syncthreads();
      for(int i = 0; i < load_iter; i++){
        int idx = col_idx + blockDim.x * i;
        if(idx < (2 * K)){
          SharedMem[idx] = *(params.ref_A.data() + idx_a_1 + idx);
          // printf("batch_idx: %d, col_idx: %d, global: (%f), shared: (%f)\n", batch_idx, col_idx, *(params.ref_A.data() + idx_a_1 + col_idx), SharedMem[col_idx]);
        }
      }
      __syncthreads();
      
      #pragma unroll 128
      for(int k = 0; k < K; k++){
        // float a1 = *(params.ref_A.data() + idx_a_1 + k);
        // float a2 = *(params.ref_A.data() + idx_a_2 + k);

        float a1 = SharedMem[k];
        float a2 = SharedMem[k + K];

        // if(a1 != SharedMem[k] || a2 != SharedMem[k + K]){
        //   printf("--batch_idx: %d, col_idx: %d, k: %d, global: (%f, %f), shared: (%f, %f)\n", batch_idx, col_idx, k, a1, a2, SharedMem[k], SharedMem[k + K]);
        // }

        float b = *(params.ref_B.data() + idx_b + k * N);
        accum1 += a1 * b;

        // float b2 = *(params.ref_B.data() + idx_b + k * N);
        accum2 += a2 * b;
      }
      *(params.ref_D.data() + idx_chk_1) = accum1;
      *(params.ref_D.data() + idx_chk_2) = accum2;
    }
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