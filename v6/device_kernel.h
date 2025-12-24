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
#include <cuda/pipeline>
#include <mma.h>
// #include <cmath>
// #include "cutlass/gemm_ring_queue.h"

using namespace cooperative_groups;
using namespace nvcuda;

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
void Kernel_Batched(typename Operator::Params params, 
            int if_split_phase, int *SM_check_res, int partion, int matrix_SM, int monitored_batched_count
            // int *all_start, int *compute, int *finding, int *recompute, int *compare, int *checking
          ) {  
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage, if_split_phase, SM_check_res, partion, matrix_SM, monitored_batched_count
    // all_start, compute, finding, recompute, compare, checking
  );
  
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

/// Generic CUTLASS kernel template.
template <typename Operator>
CUTLASS_GLOBAL
void Kernel_Update(typename Operator::Params params, 
            int if_split_phase, int *SM_check_res, int partion, int matrix_SM, int monitored_batched_count
            // int *all_start, int *compute, int *finding, int *recompute, int *compare, int *checking
          ) {  
  // printf("update kernel\n");
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage, if_split_phase, SM_check_res, partion, matrix_SM, monitored_batched_count
    // all_start, compute, finding, recompute, compare, checking
  );
  
  cutlass::arch::synclog_print();
}

template <typename Operator>
CUTLASS_GLOBAL
void update_checksum(typename Operator::Params params, int matrix_SM, int batch_per_TB){
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

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  // int TB_per_batch = (batch_per_TB > 6) ? 6 : batch_per_TB;
  int TB_per_batch = 1;

  int chk_step = chk_SM * TB_per_batch;
  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  int local_smid = real_smid - matrix_SM;
  int checksum_stride = 2 * K;

  int col_idx = threadIdx.x;
  int load_iter = (int)(ceil((double)(checksum_stride)/ (double)blockDim.x));

  // int thread_group_idx = threadIdx.x / N;
  // int init_batch = (local_smid * TB_per_batch) + thread_group_idx;
  // int col_idx = threadIdx.x % N;
  // int local_col_dim = blockDim.x / batch_per_TB;

  // int load_iter = (int)(ceil((double)(checksum_stride)/ (double)local_col_dim));
  // int shared_offset = thread_group_idx * checksum_stride;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    // int batch_idx = init_batch + b_iter * chk_step; 
    if(batch_idx < params.batch_count){
      // if(threadIdx.x == 0) {
        // printf("%d, batch_per_TB: %d, smid: %d, thread_idx: %d, thread_group_idx: %d, init_batch: %d, batch idx: %d,\n", b_iter, batch_per_TB, real_smid, threadIdx.x, thread_group_idx, init_batch, batch_idx);
      // }
    
      float accum1 = 0.f;
      float accum2 = 0.f;
      
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      // int idx_a_2 = (batch_idx * params.stride_A) + m1k;
      int idx_b = (batch_idx * params.stride_B) + col_idx;

      int idx_chk_1 = (batch_idx * params.stride_D + mn) + col_idx;
      int idx_chk_2 = (batch_idx * params.stride_D) + m1n + col_idx;

      // load checksum to share memroy
      for(int i = 0; i < load_iter; i++){
        int idx = col_idx + blockDim.x * i;
        if(idx < checksum_stride){
          SharedMem[idx] = *(params.ref_A.data() + idx_a_1 + idx);
          // printf("batch_idx: %d, col_idx: %d, global: (%f), shared: (%f)\n", batch_idx, col_idx, *(params.ref_A.data() + idx_a_1 + col_idx), SharedMem[col_idx]);
        }
      }

      // load checksum to share memroy
      // if(thread_group_idx < TB_per_batch){
      //   for(int i = 0; i < load_iter; i++){
      //     int idx = col_idx + local_col_dim * i;
      //     if(idx < checksum_stride){
      //       SharedMem[idx + shared_offset] = *(params.ref_A.data() + idx_a_1 + idx);
      //       // printf("batch_idx: %d, col_idx: %d, global: (%f), shared: (%f)\n", batch_idx, col_idx, *(params.ref_A.data() + idx_a_1 + col_idx), SharedMem[col_idx]);
      //     }
      //   }
      // }      
      __syncthreads();
      
      if(col_idx < N){
      // if(thread_group_idx < TB_per_batch){
        #pragma unroll 128
        for(int k = 0; k < K; k++){
          // float x = *(params.ref_A.data() + idx_a_1 + k);
          // float y = *(params.ref_A.data() + idx_a_2 + k);

          float a1 = SharedMem[k];
          float a2 = SharedMem[k + K];
  
          // float a1 = SharedMem[k + shared_offset];
          // float a2 = SharedMem[k + K + shared_offset];

          // if(x != a1 || y != a2){
          //   printf("--batch_idx: %d, col_idx: %d, k: %d, global: (%f, %f), shared: (%f, %f)\n", batch_idx, col_idx, k, x, y, a1, a2);
          // }

          float b = *(params.ref_B.data() + idx_b + k * N);
          
          accum1 += a1 * b;
          accum2 += a2 * b;
        }
        *(params.ref_D.data() + idx_chk_1) = accum1;
        *(params.ref_D.data() + idx_chk_2) = accum2;
      }
      __syncthreads();
    }
  } 
}

template <typename Operator>
CUTLASS_GLOBAL
void update_checksum_v2(typename Operator::Params params, int matrix_SM, int batch_per_TB){
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

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  int TB_per_batch = (batch_per_TB > 6) ? 6 : batch_per_TB;

  int chk_step = chk_SM * TB_per_batch;
  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  int local_smid = real_smid - matrix_SM;
  int checksum_stride = 2 * K;

  int thread_group_idx = threadIdx.x / N;
  int init_batch = (local_smid * TB_per_batch) + thread_group_idx;
  int col_idx = threadIdx.x % N;
  int local_col_dim = blockDim.x / batch_per_TB;

  int load_iter = (int)(ceil((double)(checksum_stride)/ (double)local_col_dim));
  int shared_offset = thread_group_idx * checksum_stride;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = init_batch + b_iter * chk_step; 
    if(batch_idx < params.batch_count){
      // if(threadIdx.x == 0) {
        // printf("%d, batch_per_TB: %d, smid: %d, thread_idx: %d, thread_group_idx: %d, init_batch: %d, batch idx: %d,\n", b_iter, batch_per_TB, real_smid, threadIdx.x, thread_group_idx, init_batch, batch_idx);
      // }
    
      float accum1 = 0.f;
      float accum2 = 0.f;
      
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      // int idx_a_2 = (batch_idx * params.stride_A) + m1k;
      int idx_b = (batch_idx * params.stride_B) + col_idx;

      int idx_chk_1 = (batch_idx * params.stride_D + mn) + col_idx;
      int idx_chk_2 = (batch_idx * params.stride_D) + m1n + col_idx;

      // load checksum to share memroy
      if(thread_group_idx < TB_per_batch){
        for(int i = 0; i < load_iter; i++){
          int idx = col_idx + local_col_dim * i;
          if(idx < checksum_stride){
            SharedMem[idx + shared_offset] = *(params.ref_A.data() + idx_a_1 + idx);
            // printf("batch_idx: %d, col_idx: %d, global: (%f), shared: (%f)\n", batch_idx, col_idx, *(params.ref_A.data() + idx_a_1 + col_idx), SharedMem[col_idx]);
          }
        }
      }      
      __syncthreads();
      
      if(thread_group_idx < TB_per_batch){
        #pragma unroll 128
        for(int k = 0; k < K; k++){
          float a1 = SharedMem[k + shared_offset];
          float a2 = SharedMem[k + K + shared_offset];

          float b = *(params.ref_B.data() + idx_b + k * N);
          
          accum1 += a1 * b;
          accum2 += a2 * b;
        }
        *(params.ref_D.data() + idx_chk_1) = accum1;
        *(params.ref_D.data() + idx_chk_2) = accum2;
      }
      __syncthreads();
    }
  } 
}

template <typename Operator, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_v3(typename Operator::Params params, int matrix_SM, int TB_per_batch, int num_sms){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  if(real_smid < matrix_SM) return;
  
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  // return gemm SM (96)
  // int matrix_SM = 128;
  int chk_SM = num_sms - matrix_SM;
  int tid = threadIdx.x;

  extern __shared__ Dtype SharedMem[];
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  // int TB_per_batch = (batch_per_TB > 6) ? 6 : batch_per_TB;
  // int TB_per_batch = 1;

  int chk_step = chk_SM * TB_per_batch;
  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  int local_smid = real_smid - matrix_SM;
  int checksum_stride = 2 * K;

  int col_idx = tid;
  int load_iter = (int)(ceil((double)(checksum_stride)/ (double)blockDim.x));
  
  int thread_group_idx = tid / N;
  int local_col_idx = tid % N;
  // int local_col_dim = blockDim.x / batch_per_TB;

  int init_batch = (local_smid * TB_per_batch) + thread_group_idx;
  int start_bid = local_smid * TB_per_batch;


  int shared_offset = thread_group_idx * checksum_stride;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    // load checksum to share memroy
    // int load_init_batch_idx = local_smid + b_iter * chk_step; 
    int load_init_batch_idx = start_bid + b_iter * chk_step; 
    for(int t = 0; t < TB_per_batch; t++){
      int load_batch_idx = load_init_batch_idx + t;
      if(load_batch_idx < params.batch_count){
        int load_offset = t * checksum_stride;
        int idx_a = (load_batch_idx * params.stride_A) + mk;
        for(int i = 0; i < load_iter; i++){
          int idx = col_idx + blockDim.x * i;
          if(idx < checksum_stride){
            SharedMem[idx + load_offset] = *(params.ref_A.data() + idx_a + idx);
          }
        }
      }
      // if(load_batch_idx < params.batch_count && col_idx < K){
      //   int load_offset = t * checksum_stride;
      //   int idx_a = (load_batch_idx * params.stride_A) + mk;
      //   int idx = col_idx + K;
      //   SharedMem[col_idx + load_offset] = *(params.ref_A.data() + idx_a + col_idx);
      //   SharedMem[idx + load_offset] = *(params.ref_A.data() + idx_a + idx);
      // }
    }
    __syncthreads();
    
    // update checksum
    int batch_idx = init_batch + b_iter * chk_step; 
    if(batch_idx < params.batch_count && thread_group_idx < TB_per_batch){
      // Dtype accum1 = static_cast<Dtype>(0.f);
      // Dtype accum2 = static_cast<Dtype>(0.f);
      float accum1 = 0.f;
      float accum2 = 0.f;
      
      int idx_b = (batch_idx * params.stride_B) + local_col_idx;

      int offset_D = batch_idx * params.stride_D;
      int idx_chk_1 = (offset_D + mn) + local_col_idx;
      int idx_chk_2 = (offset_D + m1n) + local_col_idx;
      
      int weighted_offset = K + shared_offset;
      
      #pragma unroll 128
      for(int k = 0; k < K; k++){  
        Dtype a1 = SharedMem[k + shared_offset];
        Dtype a2 = SharedMem[k + weighted_offset];

        Dtype b = *(params.ref_B.data() + idx_b + k * N);
        
        accum1 += static_cast<float>(a1 * b);
        accum2 += static_cast<float>(a2 * b);
      }
      *(params.ref_D.data() + idx_chk_1) = static_cast<Dtype>(accum1);
      *(params.ref_D.data() + idx_chk_2) = static_cast<Dtype>(accum2);
    }
    __syncthreads();
  } 
}

template <typename Operator, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_wmma(typename Operator::Params params, int matrix_SM, int TB_per_batch, int num_sms){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  if(real_smid < matrix_SM) return;
  
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  // return gemm SM (96)
  // int matrix_SM = 128;
  int chk_SM = num_sms - matrix_SM;
  int tid = threadIdx.x;
  int warp_id = tid / 32;

  extern __shared__ Dtype SharedMem[];
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();

  int checksum_stride = 16 * K;
  float *smem_base = reinterpret_cast<float*>(SharedMem + TB_per_batch * checksum_stride);

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  int warps_per_batch = N / 16;

  int chk_step = chk_SM * TB_per_batch;
  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  int local_smid = real_smid - matrix_SM;

  int col_idx = tid;
  int load_iter = (int)(ceil((double)(checksum_stride)/ (double)blockDim.x));
  
  // int thread_group_idx = tid / (2 * N);
  int local_tid = tid % (2 * N);
  int warp_level_tid = tid % 32;

  int warp_group_idx = warp_id / warps_per_batch;
  int local_warp_idx = warp_id % warps_per_batch;

  // int local_col_dim = blockDim.x / batch_per_TB;

  int init_batch = (local_smid * TB_per_batch) + warp_group_idx;
  int start_bid = local_smid * TB_per_batch;

  // int shared_offset = thread_group_idx * checksum_stride;
  int shared_offset = warp_group_idx * checksum_stride;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_acc;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    // load checksum to share memroy
    // int load_init_batch_idx = local_smid + b_iter * chk_step; 
    int load_init_batch_idx = start_bid + b_iter * chk_step; 
    for(int t = 0; t < TB_per_batch; t++){
      int load_batch_idx = load_init_batch_idx + t;
      if(load_batch_idx < params.batch_count){
        int load_offset = t * checksum_stride;
        int idx_a = (load_batch_idx * params.stride_A) + mk;
        for(int i = 0; i < load_iter; i++){
          int idx = col_idx + blockDim.x * i;
          if(idx < checksum_stride){
            SharedMem[idx + load_offset] = *(params.ref_A.data() + idx_a + idx);
          }
        }
      }
    }
    __syncthreads();
    
    // update checksum
    int batch_idx = init_batch + b_iter * chk_step; 
    if(batch_idx < params.batch_count && warp_group_idx < TB_per_batch){
      wmma::fill_fragment(c_acc, 0.0f);

      __nv_bfloat16 *a = reinterpret_cast<__nv_bfloat16*>(SharedMem + shared_offset);
      
      int idx_b = batch_idx * params.stride_B;
      __nv_bfloat16 *b = reinterpret_cast<__nv_bfloat16*>(params.ref_B.data() + idx_b);

      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;

      #pragma unroll
      for(int k = 0; k < K; k +=16){
        // Load A
        wmma::load_matrix_sync(a_frag, (a + k), K);
        // Load B
        wmma::load_matrix_sync(b_frag, (b + (local_warp_idx * 16) + (k * N)), N);
        // MMA
        wmma::mma_sync(c_acc, a_frag, b_frag, c_acc);
      }

      // Store
      int warp_offset_c = (local_warp_idx * 16) + warp_group_idx * (16 * N);
      wmma::store_matrix_sync((smem_base + warp_offset_c), c_acc, N, wmma::mem_row_major);
      
      // __syncthreads();
      // int idx_chk = (batch_idx * params.stride_D + mn) + local_tid;
      // float val_f32 = smem_base[warp_group_idx * (16 * N) + local_tid];
      // *(params.ref_D.data() + idx_chk) = static_cast<Dtype>(val_f32);

      int frag_row = warp_level_tid / 16;
      int frag_col = warp_level_tid % 16 + local_warp_idx * 16;
      int seme_idx = frag_col + frag_row * N + warp_group_idx * (16 * N);
      int d_idx = frag_col + frag_row * N + batch_idx * params.stride_D + mn;
      *(params.ref_D.data() + d_idx) = static_cast<Dtype>(smem_base[seme_idx]);
    }
    __syncthreads();
  } 
}


template <typename Operator, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_v3_T(typename Operator::Params params, int matrix_SM, int TB_per_batch){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  if(real_smid < matrix_SM) return;
  
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  // return gemm SM (96)
  // int matrix_SM = 128;
  int chk_SM = 132 - matrix_SM;
  int tid = threadIdx.x;

  extern __shared__ Dtype SharedMem[];
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  // int TB_per_batch = (batch_per_TB > 6) ? 6 : batch_per_TB;
  // int TB_per_batch = 1;

  int chk_step = chk_SM * TB_per_batch;
  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  int local_smid = real_smid - matrix_SM;
  int checksum_stride = 2 * K;

  int col_idx = tid;
  int load_iter = (int)(ceil((double)(checksum_stride)/ (double)blockDim.x));
  
  int thread_group_idx = tid / N;
  int local_col_idx = tid % N;
  // int local_col_dim = blockDim.x / batch_per_TB;

  int start_bid = local_smid * TB_per_batch;
  int init_batch = start_bid + thread_group_idx;

  int shared_offset = thread_group_idx * checksum_stride;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    // load checksum to share memroy
    int load_init_batch_idx = start_bid + b_iter * chk_step; 
    for(int t = 0; t < TB_per_batch; t++){
      int load_batch_idx = load_init_batch_idx + t;
      if(load_batch_idx < params.batch_count){
        int load_offset = t * checksum_stride;
        int idx_a = (load_batch_idx * params.stride_A) + mk;
        for(int i = 0; i < load_iter; i++){
          int idx = col_idx + blockDim.x * i;
          if(idx < checksum_stride){
            SharedMem[idx + load_offset] = *(params.ref_A.data() + idx_a + idx);
          }
        }
      }
      // if(load_batch_idx < params.batch_count && col_idx < K){
      //   int load_offset = t * checksum_stride;
      //   int idx_a = (load_batch_idx * params.stride_A) + mk;
      //   int idx = col_idx + K;
      //   SharedMem[col_idx + load_offset] = *(params.ref_A.data() + idx_a + col_idx);
      //   SharedMem[idx + load_offset] = *(params.ref_A.data() + idx_a + idx);
      // }
    }
    __syncthreads();
    
    // update checksum
    int batch_idx = init_batch + b_iter * chk_step;
    // if(threadIdx.x == 0) {
    //   printf("%d, batch_per_TB: %d, smid: %d, thread_idx: %d, thread_group_idx: %d, init_load_bach: %d, init_batch: %d, batch idx: %d, \n", 
    //           b_iter, TB_per_batch, real_smid, threadIdx.x, thread_group_idx, load_init_batch_idx, init_batch, batch_idx);
    // }
    if(batch_idx < params.batch_count 
      // && thread_group_idx < TB_per_batch
    ){
      Dtype accum1 = static_cast<Dtype>(0.f);
      Dtype accum2 = static_cast<Dtype>(0.f);
      
      // load B in column-major
      int idx_b = (batch_idx * params.stride_B) + local_col_idx * K;

      int offset_D = batch_idx * params.stride_D;
      int idx_chk_1 = (offset_D + mn) + local_col_idx;
      int idx_chk_2 = (offset_D + m1n) + local_col_idx;
      
      int weighted_offset = K + shared_offset;
      
      #pragma unroll 128
      for(int k = 0; k < K; k++){  
        Dtype a1 = SharedMem[k + shared_offset];
        Dtype a2 = SharedMem[k + weighted_offset];
        
        // load B in column-major
        Dtype b = *(params.ref_B.data()+ idx_b + k);
        
        accum1 += a1 * b;
        accum2 += a2 * b;
      }
      *(params.ref_D.data() + idx_chk_1) = accum1;
      *(params.ref_D.data() + idx_chk_2) = accum2;
    }
    __syncthreads();
  } 
}

template <typename Operator, int tiled_K, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_v4_T(typename Operator::Params params, int matrix_SM){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  int chk_SM = 132 - matrix_SM;

  int tid = threadIdx.x;
  int blockdim = blockDim.x;

  extern __shared__ Dtype SharedMem[];
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  int checksum_stride = 2 * K;

  // shared memory for A
  Dtype* As = SharedMem;  
  // shared memory for B
  Dtype* Bs = As + checksum_stride; 

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  // int TB_per_batch = (batch_per_TB > 6) ? 6 : batch_per_TB;
  int TB_per_batch = 1;

  int chk_step = chk_SM * TB_per_batch;
  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  int local_smid = real_smid - matrix_SM;

  // int loadA_iter = (int)(ceil((double)(checksum_stride)/ (double)blockdim));

  // int loadB_step = (int)(ceil((double)(blockdim)/ (double)tiled_K));
  int loadB_iter = (int)(ceil((double)(tiled_K * N)/ (double)blockdim));

  int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  // int tiledB_col = tid / tiled_K;
  // int tiledB_row = tid % tiled_K;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    // int batch_idx = init_batch + b_iter * chk_step; 
    if(batch_idx < params.batch_count){          
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      int stride_b = (batch_idx * params.stride_B);

      // load checksum to share memroy
      if(tid < checksum_stride){
        As[tid] = *(params.ref_A.data() + idx_a_1 + tid);
      }
      __syncthreads();

      Dtype accum1 = 0.f;
      Dtype accum2 = 0.f;

      // for(int tile_row = 0; tile_row < K; tile_row += tiled_K){
      for(int tile_i = 0; tile_i < tiled_iter; tile_i++){
        // if(threadIdx.x == 0) {
        //   printf("%d, batch_per_TB: %d, smid: %d, thread_idx: %d, batch idx: %d, tiled_N: %d, loadB_step: %d, tile_col: %d\n", 
        //           b_iter, batch_per_TB, real_smid, threadIdx.x, batch_idx, tiled_N, loadB_step, tile_col);
        // }

        // load tiled B to shared memory
        #pragma unroll
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;

          int B_row = shared_row + tiled_K * tile_i;
          int B_col = shared_col;

          if(B_row < K && B_col < N){
             Bs[shared_idx] = *(params.ref_B.data()+ stride_b + B_row + B_col * K);
          }
        }
        __syncthreads();
        
        // if(tid < N){
        int k_b = tid * tiled_K;
        #pragma unroll tiled_K
        for(int k = 0; k < tiled_K; k++){
          // int k_a = k + tile_row;
          int k_a = k + tile_i * tiled_K;
          if(k_a < K){
            Dtype a1 = As[k_a];
            Dtype a2 = As[k_a + K];

            Dtype b = Bs[k + k_b];
            // float b = 1;
            
            accum1 += a1 * b;
            accum2 += a2 * b;
          }
        }
        // }
        __syncthreads();
      }
      int idx_chk_1 = (batch_idx * params.stride_D + mn) + (tid);
      int idx_chk_2 = (batch_idx * params.stride_D + m1n) + (tid);

      *(params.ref_D.data() + idx_chk_1) = accum1;
      *(params.ref_D.data() + idx_chk_2) = accum2;
    }
    // __syncthreads();
  } 
}


template <typename Operator, int tiled_K, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_v5_T(typename Operator::Params params, int matrix_SM, int *SM_local_blkIdx){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;

  if(real_smid < matrix_SM) return;

  extern __shared__ Dtype SharedMem[];

  // local block idx for each SM
  Dtype *local_block_id = SharedMem;
  int temp;
  if (threadIdx.x == 0) {
      temp = atomicAdd(&SM_local_blkIdx[real_smid], 1);
      // printf("smid: %d, local_block_id: %d\n", real_smid, temp);
      *local_block_id = (Dtype) temp;
  }
  __syncthreads();

  // int chk_SM = 132 - matrix_SM;

  int blockdim = blockDim.x;
  int tid = threadIdx.x;
  int col_offset = blockdim * (*local_block_id);
  int global_col_idx = tid + col_offset;

  // printf("smid: %d, local_block_id: %f, tid: %d, gtid: %d\n", real_smid, (*local_block_id), tid, global_col_idx);
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  int checksum_stride = 2 * K;

  // shared memory for A
  Dtype* As = SharedMem + 1;  
  // shared memory for B
  Dtype* Bs = As + checksum_stride; 

  int mk = M * K;
  int mn = M * N;
  int m1n = (M + 1) * N;
  
  int chk_step = 132 - matrix_SM;
  int local_smid = real_smid - matrix_SM;

  // int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  int loadB_iter = (int)(ceil((double)(tiled_K * (N / 2))/ (double)blockdim));
  // int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  // if(tid == 0){
  //   printf("%d, %d, %d\n", chk_iter, loadB_iter, tiled_iter);
  // }

  int chk_iter = params.batch_count / chk_step;
  int tiled_iter = K / tiled_K;
  // int loadB_iter = tiled_K;

  // printf("%d, %d\n", loadB_iter, loadB_iter2);

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    if(batch_idx < params.batch_count){          
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      int stride_b = (batch_idx * params.stride_B);

      // load checksum to share memroy
      if(tid < checksum_stride){
        As[tid] = *(params.ref_A.data() + idx_a_1 + tid);
      }
      __syncthreads();

      Dtype accum1 = 0.f;
      Dtype accum2 = 0.f;

      // for(int tile_row = 0; tile_row < K; tile_row += tiled_K){
      for(int tile_i = 0; tile_i < tiled_iter; tile_i++){
        // load tiled B to shared memory
        #pragma unroll
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;

          int B_row = shared_row + tiled_K * tile_i;
          int B_col = shared_col + col_offset;

          if(B_row < K && B_col < N){
             Bs[shared_idx] = *(params.ref_B.data()+ stride_b + B_row + B_col * K);
          }
        }
        __syncthreads();
        
        int k_b = tid * tiled_K;
        #pragma unroll tiled_K
        for(int k = 0; k < tiled_K; k++){
          // int k_a = k + tile_row;
          int k_a = k + tile_i * tiled_K;
          if(k_a < K){
            Dtype a1 = As[k_a];
            Dtype a2 = As[k_a + K];

            Dtype b = Bs[k + k_b];
            // float b = 1;
            
            accum1 += a1 * b;
            accum2 += a2 * b;
          }
        }
        __syncthreads();
      }
      int idx_chk_1 = (batch_idx * params.stride_D + mn) + (global_col_idx);
      int idx_chk_2 = (batch_idx * params.stride_D + m1n) + (global_col_idx);

      *(params.ref_D.data() + idx_chk_1) = accum1;
      *(params.ref_D.data() + idx_chk_2) = accum2;
    }
    // __syncthreads();
  } 
}

template <typename Operator, int tiled_K, int num_stages, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_v6_T(typename Operator::Params params, int matrix_SM){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  int tid = threadIdx.x;
  int blockdim = blockDim.x;

  extern __shared__ Dtype SharedMem[];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  int checksum_stride = 2 * K;
  int semeB_stride = tiled_K * N;

  // shared memory for A
  Dtype* As = SharedMem;  
  // shared memory for B
  Dtype* Bs = As + checksum_stride;

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  int chk_step = 132 - matrix_SM;
  int local_smid = real_smid - matrix_SM;
  // int chk_step = chk_SM * TB_per_batch;

  // int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  // int loadB_iter = (int)(ceil((double)(semeB_stride)/ (double)blockdim));
  // int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  int chk_iter = params.batch_count / chk_step;
  int loadB_iter = semeB_stride / blockdim;
  int tiled_iter = K / tiled_K;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    // int batch_idx = init_batch + b_iter * chk_step; 
    if(batch_idx < params.batch_count){          
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      int stride_b = (batch_idx * params.stride_B);

      // load checksum to share memroy
      if(tid < checksum_stride){
        As[tid] = *(params.ref_A.data() + idx_a_1 + tid);
      }
      __syncthreads();

      Dtype accum1 = 0.f;
      Dtype accum2 = 0.f;

      // load all stages
      for(int stage = 0; stage < num_stages; stage++){
        pipe.producer_acquire();
        // load B to shared memory
        Dtype *buf = Bs + stage * semeB_stride;
        // #pragma unroll
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;
          int B_row = shared_row + tiled_K * stage;
          int B_col = shared_col;
          cuda::memcpy_async(&buf[shared_idx], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipe);
          
          // // avoid shared memory bank conflicts
          // int idx = tid + i * blockdim;
          // int shared_row = idx / tiled_K;
          // int shared_col = idx % tiled_K;
          // // transpose row and col
          // int B_row = shared_col + tiled_K * stage;
          // int B_col = shared_row;
          // cuda::memcpy_async(&buf[shared_row + shared_col * N], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipe);
        }
        pipe.producer_commit();
      }

      int stage = 0;
      for(int tile_i = 0; tile_i < tiled_iter; tile_i++){
        cuda::pipeline_consumer_wait_prior<num_stages - 1>(pipe);
        __syncthreads();

        Dtype *buf = Bs + (stage) * semeB_stride;
        int k_b = tid * tiled_K;
        int k_a_stride = tile_i * tiled_K;

        // computation
        #pragma unroll tiled_K
        for(int k = 0; k < tiled_K; k++){
          // int k_a = k + tile_row;
          // int k_a = ;
          // if(k_a < K){
            Dtype a1 = As[k + k_a_stride];
            Dtype a2 = As[k + k_a_stride + K];

            Dtype b = buf[k + k_b];
            // Dtype b = buf[k * N + tid];
            // Dtype b = 1;
            
            accum1 += a1 * b;
            accum2 += a2 * b;
          // }
        }
        __syncthreads();
        pipe.consumer_release();

        pipe.producer_acquire();
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;
          int B_row = shared_row + k_a_stride;
          int B_col = shared_col;
          cuda::memcpy_async(&buf[shared_idx], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipe);
        
          // int idx = tid + i * blockdim;
          // int shared_row = idx / tiled_K;
          // int shared_col = idx % tiled_K;
          // // transpose row and col
          // int B_row = shared_col + tiled_K * tile_i;
          // int B_col = shared_row;
          // cuda::memcpy_async(&buf[shared_row + shared_col * N], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipe);
        }
        pipe.producer_commit();

        stage = (stage + 1) % num_stages;
      }

      int idx_chk_1 = (batch_idx * params.stride_D + mn) + (tid);
      int idx_chk_2 = (batch_idx * params.stride_D + m1n) + (tid);

      *(params.ref_D.data() + idx_chk_1) = accum1;
      *(params.ref_D.data() + idx_chk_2) = accum2;
    }
    // __syncthreads();
  } 
}


template <typename Operator, int tiled_K, int num_stages, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_v7_T(typename Operator::Params params, int matrix_SM){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  int tid = threadIdx.x;
  int blockdim = blockDim.x;

  extern __shared__ Dtype SharedMem[];

  auto group = cooperative_groups::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  int checksum_stride = 2 * K;
  int semeB_stride = tiled_K * N;

  // shared memory for A
  Dtype* As = SharedMem;  
  // shared memory for B
  Dtype* Bs = As + checksum_stride;

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  int chk_step = 132 - matrix_SM;
  int local_smid = real_smid - matrix_SM;
  // int chk_step = chk_SM * TB_per_batch;

  // int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  // int loadB_iter = (int)(ceil((double)(semeB_stride)/ (double)blockdim));
  // int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  int chk_iter = params.batch_count / chk_step;
  int loadB_iter = semeB_stride / blockdim;
  int tiled_iter = K / tiled_K;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    if(batch_idx < params.batch_count){          
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      int stride_b = (batch_idx * params.stride_B);

      // load checksum to share memroy
      if(tid < checksum_stride){
        As[tid] = *(params.ref_A.data() + idx_a_1 + tid);
      }
      __syncthreads();

      Dtype accum1 = 0.f;
      Dtype accum2 = 0.f;

      // load first stage
      pipeline.producer_acquire();
      for(int i = 0; i < loadB_iter; i++){
        int shared_idx = tid + i * blockdim;
        int shared_row = shared_idx % tiled_K;
        int shared_col = shared_idx / tiled_K;
        int B_row = shared_row;
        int B_col = shared_col;
        cuda::memcpy_async(&Bs[shared_idx], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);

        // // avoid shared memory bank conflicts
        // int idx = tid + i * blockdim;
        // int shared_row = idx / tiled_K;
        // int shared_col = idx % tiled_K;
        // // transpose row and col
        // int B_row = shared_col;
        // int B_col = shared_row;
        // cuda::memcpy_async(&Bs[shared_row + shared_col * N], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
      }
      pipeline.producer_commit();

      for(int tile_i = 1; tile_i < tiled_iter; tile_i++){
        // load second stage
        pipeline.producer_acquire();
        Dtype *buf = Bs + (tile_i % num_stages) * semeB_stride;
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;
          int B_row = shared_row + tiled_K * tile_i;
          int B_col = shared_col;
          cuda::memcpy_async(&buf[shared_idx], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);

          // // avoid shared memory bank conflicts
          // int idx = tid + i * blockdim;
          // int shared_row = idx / tiled_K;
          // int shared_col = idx % tiled_K;
          // // transpose row and col
          // int B_row = shared_col + tiled_K * tile_i;
          // int B_col = shared_row;
          // cuda::memcpy_async(&buf[shared_row + shared_col * N], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
        }
        pipeline.producer_commit();
        pipeline.consumer_wait();
        
        // computation
        buf = Bs + ((tile_i - 1) % num_stages) * semeB_stride;
        int k_b = tid * tiled_K;
        int k_a_stride = (tile_i - 1) * tiled_K;

        #pragma unroll tiled_K
        for(int k = 0; k < tiled_K; k++){
          Dtype a1 = As[k + k_a_stride];
          Dtype a2 = As[k + k_a_stride + K];

          Dtype b = buf[k + k_b];
          // Dtype b = buf[k * N + tid];
          // Dtype b = 1;
          
          accum1 += a1 * b;
          accum2 += a2 * b;
        }
        pipeline.consumer_release();
      }

      pipeline.consumer_wait();
      // last stage computation
      Dtype *buf = Bs + ((tiled_iter - 1) % num_stages) * semeB_stride;
      int k_b = tid * tiled_K;
      int k_a_stride = (tiled_iter - 1) * tiled_K;
      
      #pragma unroll tiled_K
      for(int k = 0; k < tiled_K; k++){
        Dtype a1 = As[k + k_a_stride];
        Dtype a2 = As[k + k_a_stride + K];

        Dtype b = buf[k + k_b];
        // Dtype b = buf[k * N + tid];
        // Dtype b = 1;
        
        accum1 += a1 * b;
        accum2 += a2 * b;
      }
      pipeline.consumer_release();

      // 
      __syncthreads();

      int idx_chk_1 = (batch_idx * params.stride_D + mn) + (tid);
      int idx_chk_2 = (batch_idx * params.stride_D + m1n) + (tid);
      *(params.ref_D.data() + idx_chk_1) = accum1;
      *(params.ref_D.data() + idx_chk_2) = accum2;
    }
  } 
}

template <typename Operator, int tiled_K, int num_stages, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_v8_T(typename Operator::Params params, int matrix_SM, int monitored_batched_count, int num_sms){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  int tid = threadIdx.x;
  int blockdim = blockDim.x;

  extern __shared__ Dtype SharedMem[];

  auto group = cooperative_groups::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  int checksum_stride = 2 * K;
  int semeB_stride = tiled_K * N;
  int stageB_stride = (tiled_K+1) * N;

  // shared memory for A
  Dtype* As = SharedMem;  
  // shared memory for B
  Dtype* Bs = As + checksum_stride;

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  int chk_step = num_sms - matrix_SM;
  int local_smid = real_smid - matrix_SM;
  // int chk_step = chk_SM * TB_per_batch;

  // int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  // int loadB_iter = (int)(ceil((double)(semeB_stride)/ (double)blockdim));
  // int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  // int chk_iter = params.batch_count / chk_step;
  int chk_iter = monitored_batched_count / chk_step;
  int loadB_iter = semeB_stride / blockdim;
  int tiled_iter = K / tiled_K;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    // if(batch_idx < params.batch_count){
    if(batch_idx < monitored_batched_count){                    
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      int stride_b = (batch_idx * params.stride_B);

      // load checksum to share memroy
      if(tid < checksum_stride){
        As[tid] = *(params.ref_A.data() + idx_a_1 + tid);
      }
      __syncthreads();

      // Dtype accum1 = static_cast<Dtype>(0.f);
      // Dtype accum2 = static_cast<Dtype>(0.f);
      float accum1 = 0.f;
      float accum2 = 0.f;

      // load first stage
      pipeline.producer_acquire();
      for(int i = 0; i < loadB_iter; i++){
        int shared_idx = tid + i * blockdim;
        int shared_row = shared_idx % tiled_K;
        int shared_col = shared_idx / tiled_K;
        int B_row = shared_row;
        int B_col = shared_col;
        cuda::memcpy_async(&Bs[shared_row + shared_col * (tiled_K+1)], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
      }
      pipeline.producer_commit();

      for(int tile_i = 1; tile_i < tiled_iter; tile_i++){
        // load second stage
        pipeline.producer_acquire();
        Dtype *buf = Bs + (tile_i % num_stages) * stageB_stride;
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;
          int B_row = shared_row + tiled_K * tile_i;
          int B_col = shared_col;
          cuda::memcpy_async(&buf[shared_row + shared_col * (tiled_K+1)], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
        }
        pipeline.producer_commit();
        pipeline.consumer_wait();
        
        // computation
        buf = Bs + ((tile_i - 1) % num_stages) * stageB_stride;
        int k_b = tid * (tiled_K + 1);
        int k_a_stride = (tile_i - 1) * tiled_K;

        #pragma unroll tiled_K
        for(int k = 0; k < tiled_K; k++){
          Dtype a1 = As[k + k_a_stride];
          Dtype a2 = As[k + k_a_stride + K];

          Dtype b = buf[k + k_b];
          // Dtype b = 1;
          
          accum1 += static_cast<float>(a1 * b);
          accum2 += static_cast<float>(a2 * b);
        }
        pipeline.consumer_release();
      }

      pipeline.consumer_wait();
      // last stage computation
      Dtype *buf = Bs + ((tiled_iter - 1) % num_stages) * stageB_stride;
      int k_b = tid * (tiled_K+1);
      int k_a_stride = (tiled_iter - 1) * tiled_K;
      
      #pragma unroll tiled_K
      for(int k = 0; k < tiled_K; k++){
        Dtype a1 = As[k + k_a_stride];
        Dtype a2 = As[k + k_a_stride + K];

        Dtype b = buf[k + k_b];
        // Dtype b = 1;
        
        accum1 += static_cast<float>(a1 * b);
        accum2 += static_cast<float>(a2 * b);
      }
      pipeline.consumer_release();

      // 
      __syncthreads();

      int idx_chk_1 = (batch_idx * params.stride_D + mn) + (tid);
      int idx_chk_2 = (batch_idx * params.stride_D + m1n) + (tid);
      *(params.ref_D.data() + idx_chk_1) = static_cast<Dtype>(accum1);
      *(params.ref_D.data() + idx_chk_2) = static_cast<Dtype>(accum2);
    }
  } 
}


template <typename Operator, int tiled_K, int num_stages, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_T_wmma(typename Operator::Params params, int matrix_SM, int monitored_batched_count, int num_sms){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  int tid = threadIdx.x;
  int blockdim = blockDim.x;
  int warp_id = tid / 32;

  extern __shared__ Dtype SharedMem[];

  auto group = cooperative_groups::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);

  // Accumulator Frag
  // wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  // wmma::fill_fragment(c_frag, 0.0f);
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_acc[2];
  // wmma::fill_fragment(c_acc[0], 0.0f);
  // wmma::fill_fragment(c_acc[1], 0.0f);
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  int checksum_stride = 16 * K;
  int semeB_stride = tiled_K * N;
  int semeB_ld = tiled_K + 8;
  int stageB_stride = semeB_ld * N;

  int checksum_load_stride = 2 * K;

  // shared memory for A
  // __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(SharedMem);
  Dtype* As = SharedMem;
  // shared memory for B
  Dtype* Bs = As + checksum_stride;

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  int chk_step = num_sms - matrix_SM;
  int local_smid = real_smid - matrix_SM;
  // int chk_step = chk_SM * TB_per_batch;

  // int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  // int loadB_iter = (int)(ceil((double)(semeB_stride)/ (double)blockdim));
  // int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  // int chk_iter = params.batch_count / chk_step;
  int chk_iter = monitored_batched_count / chk_step;
  int loadB_iter = semeB_stride / blockdim;
  int tiled_iter = K / tiled_K;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    // if(batch_idx < params.batch_count){
    if(batch_idx < monitored_batched_count){                    
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      int stride_b = (batch_idx * params.stride_B);

      // load checksum to share memroy
      // for(int i = tid; i < checksum_stride; i += blockdim) {
      //     As[i] = *(params.ref_A.data() + idx_a_1 + i);
      // }
      for(int i = tid; i < checksum_load_stride; i += blockdim) {
          As[i] = *(params.ref_A.data() + idx_a_1 + i);
      }
      __syncthreads();
      
      // FIX 2: Reset Accumulator inside the loop
      wmma::fill_fragment(c_acc[0], 0.0f);
      wmma::fill_fragment(c_acc[1], 0.0f);

      // load first stage
      pipeline.producer_acquire();
      for(int i = 0; i < loadB_iter; i++){
        int shared_idx = tid + i * blockdim;
        int shared_row = shared_idx % tiled_K;
        int shared_col = shared_idx / tiled_K;
        int B_row = shared_row;
        int B_col = shared_col;
        cuda::memcpy_async(&Bs[shared_row + shared_col * semeB_ld], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
      }
      pipeline.producer_commit();

      for(int tile_i = 1; tile_i < tiled_iter; tile_i++){
        // load second stage
        pipeline.producer_acquire();
        Dtype *buf = Bs + (tile_i % num_stages) * stageB_stride;
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;
          int B_row = shared_row + tiled_K * tile_i;
          int B_col = shared_col;
          cuda::memcpy_async(&buf[shared_row + shared_col * semeB_ld], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
        }
        pipeline.producer_commit();
        pipeline.consumer_wait();
        
        // computation
        buf = Bs + ((tile_i - 1) % num_stages) * stageB_stride;
        // int k_b = tid * (tiled_K + 1);
        int k_a_stride = (tile_i - 1) * tiled_K;
        // int tile_b_stride = stageB_stride / 2;

        __nv_bfloat16 *a = reinterpret_cast<__nv_bfloat16*>(As + k_a_stride);
        __nv_bfloat16 *b = reinterpret_cast<__nv_bfloat16*>(buf);

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;

        int b_offset_0 = warp_id * (semeB_ld * 16); // Row 0, Col offset
        int b_offset_1 = b_offset_0 + (semeB_ld * 512); // Row 0, Col offset
        
        #pragma unroll
        for(int k = 0; k < tiled_K; k += 16){
          // Load A
          wmma::load_matrix_sync(a_frag, (a+k), K);

          // Compute Tile 0 (Col 0 ~ 511)
          // int b_offset_0 = warp_id * (semeB_ld * 16) + k; // Row 0, Col offset
          wmma::load_matrix_sync(b_frag, (b + b_offset_0 + k), semeB_ld);
          wmma::mma_sync(c_acc[0], a_frag, b_frag, c_acc[0]);

          // Compute Tile 1 (Col 512 ~ 1023)
          // int b_offset_1 = b_offset_0 + (semeB_ld * 512); // Row 0, Col offset
          wmma::load_matrix_sync(b_frag, (b + b_offset_1 + k), semeB_ld);
          wmma::mma_sync(c_acc[1], a_frag, b_frag, c_acc[1]);
        }
        pipeline.consumer_release();
      }

      pipeline.consumer_wait();
      // last stage computation
      Dtype *buf = (Bs + ((tiled_iter - 1) % num_stages) * stageB_stride);
      int k_a_stride = (tiled_iter - 1) * tiled_K;
      // int tile_b_stride = stageB_stride / 2;
      
      __nv_bfloat16 *a = reinterpret_cast<__nv_bfloat16*>(As + k_a_stride);
      __nv_bfloat16 *b = reinterpret_cast<__nv_bfloat16*>(buf);

      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
      
      int b_offset_0 = warp_id * (semeB_ld * 16); // Row 0, Col offset
      int b_offset_1 = b_offset_0 + (semeB_ld * 512); // Row 0, Col offset

      #pragma unroll
      for(int k = 0; k < tiled_K; k += 16){
        // Load A
        wmma::load_matrix_sync(a_frag, (a+k), K);

        // Compute Tile 0 (Col 0 ~ 511)
        // int b_offset_0 = warp_id * (semeB_ld * 16) + k; // Row 0, Col offset
        wmma::load_matrix_sync(b_frag, (b + b_offset_0 + k), semeB_ld);
        wmma::mma_sync(c_acc[0], a_frag, b_frag, c_acc[0]);

        // Compute Tile 1 (Col 512 ~ 1023)
        // int b_offset_1 = b_offset_0 + (semeB_ld * 512); // Row 0, Col offset
        wmma::load_matrix_sync(b_frag, (b + b_offset_1 + k), semeB_ld);
        wmma::mma_sync(c_acc[1], a_frag, b_frag, c_acc[1]);
      }
      
      pipeline.consumer_release();

      // 
      __syncthreads();

      // Store
      float* smem_base = reinterpret_cast<float*>(SharedMem);
      
      int warp_offset_0 = warp_id * 16;
      int warp_offset_1 = warp_offset_0 + (N / 2);
      
      wmma::store_matrix_sync((smem_base + warp_offset_0), c_acc[0], N, wmma::mem_row_major);
      wmma::store_matrix_sync((smem_base + warp_offset_1), c_acc[1], N, wmma::mem_row_major);
      
      __syncthreads();

      float val_f32_1 = smem_base[tid];
      float val_f32_2 = smem_base[tid + N];

      int idx_chk_1 = (batch_idx * params.stride_D + mn) + (tid);
      int idx_chk_2 = (batch_idx * params.stride_D + m1n) + (tid); 
      
      *(params.ref_D.data() + idx_chk_1) = static_cast<Dtype>(val_f32_1);
      *(params.ref_D.data() + idx_chk_2) = static_cast<Dtype>(val_f32_2);

      // int warp_offset_0 = warp_id * 16;
      // int warp_offset_1 = warp_offset_0 + (N / 2);
      // int idx_chk_1 = ((batch_idx * params.stride_D + mn) + warp_offset_0);
      // int idx_chk_2 = ((batch_idx * params.stride_D + mn) + warp_offset_1); 

      // float* float_D = reinterpret_cast<float*>(params.ref_D.data());

      // wmma::store_matrix_sync(float_D + idx_chk_1, c_acc[0], N, wmma::mem_row_major);
      // wmma::store_matrix_sync(float_D + idx_chk_2, c_acc[1], N, wmma::mem_row_major);

    }
  } 
}

template <typename Operator, int tiled_K, int num_stages, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_T_wmma_v2(typename Operator::Params params, int matrix_SM, int monitored_batched_count, int num_sms){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  int tid = threadIdx.x;
  int blockdim = blockDim.x;
  int warp_id = tid / 32;

  extern __shared__ Dtype SharedMem[];

  auto group = cooperative_groups::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);

  // Accumulator Frag
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_acc[2];
    
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  // int checksum_stride = 16 * K;
  int checksum_stride = 16 * tiled_K;
  int semeB_stride = tiled_K * N;
  int semeB_ld = tiled_K + 8;
  int stageB_stride = semeB_ld * N;

  int checksum_load_stride = 2 * K;

  // shared memory for A
  // __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(SharedMem);
  Dtype* As = SharedMem;
  // shared memory for B
  Dtype* Bs = As + checksum_stride * num_stages;

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  int chk_step = num_sms - matrix_SM;
  int local_smid = real_smid - matrix_SM;
  // int chk_step = chk_SM * TB_per_batch;

  // int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  // int loadB_iter = (int)(ceil((double)(semeB_stride)/ (double)blockdim));
  // int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  // int chk_iter = params.batch_count / chk_step;
  int chk_iter = monitored_batched_count / chk_step;
  int loadB_iter = semeB_stride / blockdim;
  int tiled_iter = K / tiled_K;

  size_t bytes_per_row_A = tiled_K * sizeof(Dtype);

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    // if(batch_idx < params.batch_count){
    if(batch_idx < monitored_batched_count){                    
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      int stride_b = (batch_idx * params.stride_B);

      // FIX 2: Reset Accumulator inside the loop
      wmma::fill_fragment(c_acc[0], 0.0f);
      wmma::fill_fragment(c_acc[1], 0.0f);

      // load first stage
      pipeline.producer_acquire();
      // load A
      if (tid < checksum_stride) {
        int A_col = tid % tiled_K;
        int A_row = tid / tiled_K;
        // Global Addr: Base + Row * K + current_k (0)
        Dtype* src = params.ref_A.data() + idx_a_1 + (A_col + A_row * K);
        Dtype* dst = As + tid;
        cuda::memcpy_async(dst, src, sizeof(Dtype), pipeline);
      } 
      // load B
      for(int i = 0; i < loadB_iter; i++){
        int shared_idx = tid + i * blockdim;
        int shared_row = shared_idx % tiled_K;
        int shared_col = shared_idx / tiled_K;
        int B_row = shared_row;
        int B_col = shared_col;
        cuda::memcpy_async(&Bs[shared_row + shared_col * semeB_ld], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
      }
      pipeline.producer_commit();

      for(int tile_i = 1; tile_i < tiled_iter; tile_i++){
        // load second stage
        int load_stage_idx = tile_i % num_stages;

        pipeline.producer_acquire();
        // load A
        int k_start = tile_i * tiled_K;
        int stage_a_offset = load_stage_idx * checksum_stride;
        if (tid < checksum_stride) {
          int A_col = tid % tiled_K;
          int A_row = tid / tiled_K;
          // Global Addr: Base + Row * K + current_k (0)
          Dtype* src = params.ref_A.data() + idx_a_1 + (k_start + A_col) + A_row * K;
          Dtype* dst = (As + stage_a_offset) + tid;
          cuda::memcpy_async(dst, src, sizeof(Dtype), pipeline);
        }
        // load B
        Dtype *buf = Bs + load_stage_idx * stageB_stride;
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;
          int B_row = shared_row + tiled_K * tile_i;
          int B_col = shared_col;
          cuda::memcpy_async(&buf[shared_row + shared_col * semeB_ld], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
        }
        pipeline.producer_commit();
        pipeline.consumer_wait();
        
        // computation
        int compute_stage_idx = (tile_i - 1) % num_stages;
        buf = Bs + compute_stage_idx * stageB_stride;
        // int k_b = tid * (tiled_K + 1);
        int k_a_stride = compute_stage_idx * checksum_stride;
        // int tile_b_stride = stageB_stride / 2;

        __nv_bfloat16 *a = reinterpret_cast<__nv_bfloat16*>(As + k_a_stride);
        __nv_bfloat16 *b = reinterpret_cast<__nv_bfloat16*>(buf);

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;

        int b_offset_0 = warp_id * (semeB_ld * 16); // Row 0, Col offset
        int b_offset_1 = b_offset_0 + (semeB_ld * 512); // Row 0, Col offset
        
        #pragma unroll
        for(int k = 0; k < tiled_K; k += 16){
          // Load A
          wmma::load_matrix_sync(a_frag, (a+k), tiled_K);

          // Compute Tile 0 (Col 0 ~ 511)
          // int b_offset_0 = warp_id * (semeB_ld * 16) + k; // Row 0, Col offset
          wmma::load_matrix_sync(b_frag, (b + b_offset_0 + k), semeB_ld);
          wmma::mma_sync(c_acc[0], a_frag, b_frag, c_acc[0]);

          // Compute Tile 1 (Col 512 ~ 1023)
          // int b_offset_1 = b_offset_0 + (semeB_ld * 512); // Row 0, Col offset
          wmma::load_matrix_sync(b_frag, (b + b_offset_1 + k), semeB_ld);
          wmma::mma_sync(c_acc[1], a_frag, b_frag, c_acc[1]);
        }
        pipeline.consumer_release();
      }

      pipeline.consumer_wait();
      // last stage computation
      int compute_stage_idx = (tiled_iter - 1) % num_stages;
      Dtype *buf = Bs + compute_stage_idx * stageB_stride;
      int k_a_stride = compute_stage_idx * checksum_stride;
      // int tile_b_stride = stageB_stride / 2;
      
      __nv_bfloat16 *a = reinterpret_cast<__nv_bfloat16*>(As + k_a_stride);
      __nv_bfloat16 *b = reinterpret_cast<__nv_bfloat16*>(buf);

      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
      
      int b_offset_0 = warp_id * (semeB_ld * 16); // Row 0, Col offset
      int b_offset_1 = b_offset_0 + (semeB_ld * 512); // Row 0, Col offset

      #pragma unroll
      for(int k = 0; k < tiled_K; k += 16){
        // Load A
        wmma::load_matrix_sync(a_frag, (a+k), tiled_K);

        // Compute Tile 0 (Col 0 ~ 511)
        // int b_offset_0 = warp_id * (semeB_ld * 16) + k; // Row 0, Col offset
        wmma::load_matrix_sync(b_frag, (b + b_offset_0 + k), semeB_ld);
        wmma::mma_sync(c_acc[0], a_frag, b_frag, c_acc[0]);

        // Compute Tile 1 (Col 512 ~ 1023)
        // int b_offset_1 = b_offset_0 + (semeB_ld * 512); // Row 0, Col offset
        wmma::load_matrix_sync(b_frag, (b + b_offset_1 + k), semeB_ld);
        wmma::mma_sync(c_acc[1], a_frag, b_frag, c_acc[1]);
      }
      pipeline.consumer_release();

      // 
      __syncthreads();

      // Store
      float* smem_base = reinterpret_cast<float*>(SharedMem);
      
      int warp_offset_0 = warp_id * 16;
      int warp_offset_1 = warp_offset_0 + (N / 2);
      
      wmma::store_matrix_sync((smem_base + warp_offset_0), c_acc[0], N, wmma::mem_row_major);
      wmma::store_matrix_sync((smem_base + warp_offset_1), c_acc[1], N, wmma::mem_row_major);
      
      __syncthreads();

      float val_f32_1 = smem_base[tid];
      float val_f32_2 = smem_base[tid + N];

      int idx_chk_1 = (batch_idx * params.stride_D + mn) + (tid);
      int idx_chk_2 = (batch_idx * params.stride_D + m1n) + (tid); 
      
      *(params.ref_D.data() + idx_chk_1) = static_cast<Dtype>(val_f32_1);
      *(params.ref_D.data() + idx_chk_2) = static_cast<Dtype>(val_f32_2);
    }
  } 
}



template <typename Operator, int tiled_K, int num_stages, typename Dtype>
CUTLASS_GLOBAL
void update_checksum_v9_T(typename Operator::Params params, int matrix_SM){
  // get SM id
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
  // return gemm SM (96)
  // int matrix_SM = 128;

  if(real_smid < matrix_SM) return;
  // if(threadIdx.x == 0) {
  //   printf("update smid: %d, gird size(%d, %d, %d), block size(%d, %d, %d), blk_idx: %d\n", 
  //           real_smid, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x);
  // }

  int tid = threadIdx.x;
  int blockdim = blockDim.x;
  int warpid = tid / 32;
  int warp_tid = tid % 32;

  extern __shared__ Dtype SharedMem[];

  auto group = cooperative_groups::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);
  
  // int thread_idx = threadIdx.x;
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  int checksum_stride = 2 * K;
  int semeB_stride = tiled_K * N;

  // shared memory for A
  Dtype* As = SharedMem;  
  // shared memory for B
  Dtype* Bs = As + checksum_stride;

  int mk = M * K;
  int mn = M * N;
  // int m1k =(M + 1) * K;
  int m1n = (M + 1) * N;
  
  int chk_step = 132 - matrix_SM;
  int local_smid = real_smid - matrix_SM;
  // int chk_step = chk_SM * TB_per_batch;

  // int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  // int loadB_iter = (int)(ceil((double)(semeB_stride)/ (double)blockdim));
  // int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  int chk_iter = params.batch_count / chk_step;
  int loadB_iter = semeB_stride / blockdim;
  int tiled_iter = K / tiled_K;

  for(int b_iter = 0; b_iter < chk_iter; b_iter += 1){
    int batch_idx = local_smid + b_iter * chk_step;
    if(batch_idx < params.batch_count){          
      int idx_a_1 = (batch_idx * params.stride_A) + mk;
      int stride_b = (batch_idx * params.stride_B);

      // load checksum to share memroy
      if(tid < checksum_stride){
        As[tid] = *(params.ref_A.data() + idx_a_1 + tid);
      }
      __syncthreads();

      Dtype accum1 = 0.f;
      Dtype accum2 = 0.f;

      // load first stage
      pipeline.producer_acquire();
      for(int i = 0; i < loadB_iter; i++){
        int shared_idx = tid + i * blockdim;
        int shared_row = shared_idx % tiled_K;
        int shared_col = shared_idx / tiled_K;
        int B_row = shared_row;
        int B_col = shared_col;
        cuda::memcpy_async(&Bs[shared_idx], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);

        // // avoid shared memory bank conflicts
        // int idx = tid + i * blockdim;
        // int shared_row = idx / tiled_K;
        // int shared_col = idx % tiled_K;
        // // transpose row and col
        // int B_row = shared_col;
        // int B_col = shared_row;
        // cuda::memcpy_async(&Bs[shared_row + shared_col * N], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
      }
      pipeline.producer_commit();

      for(int tile_i = 1; tile_i < tiled_iter; tile_i++){
        // load second stage
        pipeline.producer_acquire();
        Dtype *buf = Bs + (tile_i % num_stages) * semeB_stride;
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;
          int B_row = shared_row + tiled_K * tile_i;
          int B_col = shared_col;
          cuda::memcpy_async(&buf[shared_idx], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);

          // // avoid shared memory bank conflicts
          // int idx = tid + i * blockdim;
          // int shared_row = idx / tiled_K;
          // int shared_col = idx % tiled_K;
          // // transpose row and col
          // int B_row = shared_col + tiled_K * tile_i;
          // int B_col = shared_row;
          // cuda::memcpy_async(&buf[shared_row + shared_col * N], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipeline);
        }
        pipeline.producer_commit();
        pipeline.consumer_wait();
        
        // computation
        buf = Bs + ((tile_i - 1) % num_stages) * semeB_stride;
        int w_row = warpid % 2;
        int w_col = (warpid / 2) * 64 + warp_tid;
        int k_b = w_col * tiled_K;
        // int k_b = (tid) * tiled_K;
        int k_a_stride = (tile_i - 1) * tiled_K;

        #pragma unroll tiled_K
        for(int k = 0; k < tiled_K; k++){
          Dtype a = As[k + k_a_stride + w_row * K];

          Dtype b1 = buf[k + k_b];
          Dtype b2 = buf[k + k_b + 32];
          // Dtype b = buf[k * N + tid];
          // Dtype b = 1;
          
          accum1 += a * b1;
          accum2 += a * b2;
        }
        pipeline.consumer_release();
      }

      pipeline.consumer_wait();
      // last stage computation
      Dtype *buf = Bs + ((tiled_iter - 1) % num_stages) * semeB_stride;
      int w_row = warpid % 2;
      int w_col = (warpid / 2) * 64 + warp_tid;
      int k_b = w_col * tiled_K;
      // int k_b = tid * tiled_K;
      int k_a_stride = (tiled_iter - 1) * tiled_K;
      
      #pragma unroll tiled_K
      for(int k = 0; k < tiled_K; k++){
        Dtype a = As[k + k_a_stride + w_row * K];

        Dtype b1 = buf[k + k_b];
        Dtype b2 = buf[k + k_b + 32];
        // Dtype b = buf[k * N + tid];
        // Dtype b = 1;
        
        accum1 += a * b1;
        accum2 += a * b2;
      }
      pipeline.consumer_release();

      // 
      __syncthreads();

      // int idx_chk_1 = (batch_idx * params.stride_D + mn) + (tid);
      // int idx_chk_2 = (batch_idx * params.stride_D + m1n) + (tid);

      int idx_chk_1 = (batch_idx * params.stride_D + (M + w_row) * N) + (w_col);
      int idx_chk_2 = idx_chk_1 + 32;
      // int idx_chk_2 = (batch_idx * params.stride_D + (M + w_row) * N) + (w_col + 32);
      *(params.ref_D.data() + idx_chk_1) = accum1;
      *(params.ref_D.data() + idx_chk_2) = accum2;
    }
  } 
}

template<typename Operator>
CUTLASS_DEVICE
void check_phase_v3(typename Operator::Params params, int batch_idx, int col_idx, int *SM_check_res, int matrix_SM, int batch_step, int &diff, int &loc){
    int M = params.problem_size.m();
    int K = params.problem_size.k();
    int N = params.problem_size.n();
    
    float E = 1e1;
    // int loc = -1;
    float MAX = 0;
    // int diff = 0;

    // recompute checksum (no weighted, weighted)
    float dA_col_r1 = 0.f;
    float dA_col_r2 = 0.f;
    
    int start_idx = (params.stride_D * batch_idx) + col_idx;
    
    #pragma unroll 128
    for(int r = 0; r < M; r++){
      int idx = start_idx + r * N;
      float element = static_cast<float>(*(params.ref_D.data() + idx));
      
      dA_col_r1 += element;
      dA_col_r2 += static_cast<float>(r+1) * element;
    }

    // detect error
    float dA_col_1 = static_cast<float>(*(params.ref_D.data() + start_idx + (M*N)));
    float dA_col_2 = static_cast<float>(*(params.ref_D.data() + start_idx + (M+1)*N));

    float d1 = (float)(dA_col_1 - dA_col_r1);
    float d2 = (float)(dA_col_2 - dA_col_r2);
    float abs_d1 = fabs(d1);

    // printf("tid: %d, batch_idx: %d, row_idx: %d, updated: (%f, %f), recomputed: (%f, %f)\n", thread_idx, batch_idx, row_idx, dA_col_1, dA_col_2, dA_col_r1, dA_col_r2);
    
    if(abs_d1 > E){
      if(!std::isinf(d2)){
        loc = round(d2 / d1) - 1;
        float max = (dA_col_1 > dA_col_r1) ? dA_col_1 : dA_col_r1;
        float rel_err = abs_d1 / max;
        printf("[col check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) update(%f, %f) recompute(%f, %f), rel_err: %f\n", (float)d1, (float)d2, loc, dA_col_1, dA_col_2, dA_col_r1, dA_col_r2, rel_err);
        diff = 1;
      }
      else{
        MAX = 0;
				int counter = 0;
				for(int i = 0; i < N; i++) {
					if(fabs((float)*(params.ref_D.data() + start_idx + i * N)) > MAX){
						MAX = fabs((float)*(params.ref_D.data() + start_idx + i * N));
						loc = i;
					}
					if(fabs((float)*(params.ref_D.data() + start_idx + i * N)) > 1e10){
						counter++;
						if(counter > 1){
							printf("[col check]col chksum error, more than one large number. (d1 = %.6f, d2 = %.6f)\n",(float)d1, (float)d2);
							return;
						}
					}
				}
				printf("[col check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
        diff = 1;        
      }
      return;
    }
    // abs == inf
    if(std::isinf(abs_d1)){
      MAX = 0;
      int64_t counter = 0;
      for(int i = 0; i < N; i++) {
        if(fabs((float)*(params.ref_D.data() + start_idx + i * N)) > MAX){
          MAX = fabs((float)*(params.ref_D.data() + start_idx + i * N));
          loc = i;
        }
        if(std::isinf(*(params.ref_D.data() + start_idx + i * N)) || fabs((float)*(params.ref_D.data() + start_idx + i * N)) > 1e10){
          counter++;
          if(counter > 1){
            printf("[col check]Multi INFs or Large Number detected in one column.(d1 = %.6f, d2 = %.6f, iter = %d)\n", (float)d1, (float)d2, i);
            return;
          }
        }
      }
      if(counter == 0){
        printf("[col chk]No INF or Large Number found.\n");
        return;
      }
      printf("[col check]INF detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
      diff = 1;
    }
    // abs == nan
	  if(std::isnan(abs_d1)){
      int64_t counter = 0;
      for(int i = 0; i < N; i++) {
        if (std::isnan(*(params.ref_D.data() + start_idx + i * N))) {
          loc = i;
          counter++;
        }
        if(std::isinf(*(params.ref_D.data() + start_idx + i * N))){
          counter++;
        }
        if(fabs((float)*(params.ref_D.data() + start_idx + i * N)) > 1e10){
          counter++;
        }
        if(counter > 1){
          printf("[col check]Multi INF, NAN or Large Number detected in one column. (iter = %d)\n", i);
          return;
        }
      }
      printf("[col check]NAN detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
      diff = 1;
    }
  }


template <typename Operator>
CUTLASS_GLOBAL
void check_SM(typename Operator::Params params, int matrix_SM, int *SM_check_res, int TB_per_batch){
  unsigned int real_smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));

  // return update SM
  if(real_smid > (matrix_SM - 1)){
    return;
  }

  // 
  // int TB_per_batch = 1;
  int SM_per_batch = params.grid_tiled_shape.m() * params.grid_tiled_shape.n() / TB_per_batch;
  int batch_step = (int)(floor((double)matrix_SM / (double)SM_per_batch));
  int check_iter = (int)(ceil((double)params.batch_count / (double)batch_step / (double)SM_per_batch));

  int local_smid = real_smid % SM_per_batch;
  int init_batch_idx = real_smid / SM_per_batch;
  
  int checked_init_batch_idx = ((init_batch_idx + 1) % batch_step) + (local_smid / TB_per_batch) * batch_step;
  
  int col_idx = threadIdx.x % params.problem_size.n();
  int check_step = SM_per_batch * batch_step;

  // if(threadIdx.x == 0) printf("real smid: %d, local smid: %d, tid: %d, check_iter: %d, init_batch_idx: %d, checked_init_batch_idx: %d\n", real_smid, local_smid, threadIdx.x, check_iter, init_batch_idx, checked_init_batch_idx);

  for(int i = 0; i < check_iter; i += 1){
    int checked_batch_idx = checked_init_batch_idx + i * check_step;
    if(checked_batch_idx < params.batch_count){
        int diff = 0, loc = -1;
      // if(row_idx < params.problem_size.n()){
        cutlass::check_phase_v3<Operator>(params, checked_batch_idx, col_idx, SM_check_res, matrix_SM, batch_step, diff, loc);
        // if(threadIdx.x == 0) printf("iter: %d, real smid: %d, local smid: %d, check_iter: %d, init_batch_idx: %d, checked_init_batch_idx: %d, checked_batch_idx: %d\n", i, real_smid, local_smid, check_iter, init_batch_idx, checked_init_batch_idx, checked_batch_idx);
      // }
        if(diff != 0){
          // Locate corrupted SM
          int error_n_offset = col_idx / 256;
          int error_m_offset = loc / 128;
          int error_local_smid = error_m_offset + error_n_offset * params.grid_tiled_shape.m();
          int error_smid = error_local_smid + ((init_batch_idx + 1) % batch_step) * SM_per_batch;

          printf("Error detected at SM %d by checker SM %d\n", error_smid, real_smid);
      
          // record results
          // Atomic sum
          atomicAdd((SM_check_res + error_smid), diff);
        }
        __syncthreads();
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