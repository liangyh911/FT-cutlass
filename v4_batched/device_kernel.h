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
void Kernel_Batched(typename Operator::Params params, 
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

template <typename Operator>
CUTLASS_GLOBAL
void update_checksum_v3(typename Operator::Params params, int matrix_SM, int TB_per_batch){
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
      float accum1 = 0.f;
      float accum2 = 0.f;
      
      int idx_b = (batch_idx * params.stride_B) + local_col_idx;

      int offset_D = batch_idx * params.stride_D;
      int idx_chk_1 = (offset_D + mn) + local_col_idx;
      int idx_chk_2 = (offset_D + m1n) + local_col_idx;
      
      int weighted_offset = K + shared_offset;
      
      #pragma unroll 128
      for(int k = 0; k < K; k++){  
        float a1 = SharedMem[k + shared_offset];
        float a2 = SharedMem[k + weighted_offset];

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

template <typename Operator, int tiled_K, int num_stages, int smem_stages, typename Dtype>
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

  int chk_iter = (int)(ceil((double)params.batch_count / (double)chk_step));
  // int loadB_iter = (int)(ceil((double)(semeB_stride)/ (double)blockdim));
  // int tiled_iter = (int)(ceil((double)(K)/ (double)tiled_K));

  // int chk_iter = params.batch_count / chk_step;
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
        #pragma unroll
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;

          int B_row = shared_row + tiled_K * stage;
          int B_col = shared_col;

          if(B_row < K && B_col < N){
            Dtype *buf = Bs + (stage % smem_stages) * semeB_stride;
            cuda::memcpy_async(&buf[shared_idx], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipe);
            // Dtype b = 1;
            // cuda::memcpy_async(&buf[shared_idx], &b, sizeof(Dtype), pipe);
            // buf[shared_idx] = *(params.ref_B.data()+ stride_b + B_row + B_col * K);
          }
        }
        pipe.producer_commit();
      }

      int stage = 0;
      for(int tile_i = 0; tile_i < tiled_iter; tile_i++){
        cuda::pipeline_consumer_wait_prior<num_stages - 1>(pipe);
        __syncthreads();

        Dtype *buf = Bs + (stage % smem_stages) * semeB_stride;
        int k_b = tid * tiled_K;

        // computation
        #pragma unroll
        for(int k = 0; k < tiled_K; k++){
          // int k_a = k + tile_row;
          int k_a = k + tile_i * tiled_K;
          if(k_a < K){
            Dtype a1 = As[k_a];
            Dtype a2 = As[k_a + K];

            Dtype b = buf[k + k_b];
            // Dtype b = 1;
            
            accum1 += a1 * b;
            accum2 += a2 * b;
          }
        }
        __syncthreads();
        pipe.consumer_release();

        pipe.producer_acquire();
        for(int i = 0; i < loadB_iter; i++){
          int shared_idx = tid + i * blockdim;
          int shared_row = shared_idx % tiled_K;
          int shared_col = shared_idx / tiled_K;

          int B_row = shared_row + tiled_K * tile_i;
          int B_col = shared_col;

          if(B_row < K && B_col < N){
            Dtype *buf = Bs + (stage % smem_stages) * semeB_stride;
            cuda::memcpy_async(&buf[shared_idx], (params.ref_B.data()+ stride_b + B_row + B_col * K), sizeof(Dtype), pipe);
            // Dtype b = 1;
            // cuda::memcpy_async(&buf[shared_idx], &b, sizeof(Dtype), pipe);
            // buf[shared_idx] = *(params.ref_B.data()+ stride_b + B_row + B_col * K);
          }
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


template<typename Operator>
CUTLASS_DEVICE
void check_phase_v3(typename Operator::Params params, int batch_idx, int col_idx, int *SM_check_res, int matrix_SM, int batch_step, int &diff, int &loc){
  int M = params.problem_size.m();
  int K = params.problem_size.k();
  int N = params.problem_size.n();
  float E = 10;
  // int loc = -1;
  float MAX = 0;
  // int diff = 0;

  // recompute checksum (no weighted, weighted)
  float dA_col_r1 = 0.f;
  float dA_col_r2 = 0.f;
  
  int start_idx = (params.stride_D * batch_idx) + col_idx;
  
  #pragma unroll 128
  for(int r = 0; r < M; r++){
    float element = *(params.ref_D.data() + (start_idx + r * N));
    
    dA_col_r1 += element;
    dA_col_r2 += (float)(r+1) * element;
  }

  // detect error
  float dA_col_1 = *(params.ref_D.data() + (start_idx + (M*N)));
  float dA_col_2 = *(params.ref_D.data() + (start_idx + (M+1)*N));

  float d1 = (float)(dA_col_1 - dA_col_r1);
  float d2 = (float)(dA_col_2 - dA_col_r2);
  float abs_d1 = fabs(d1);

  // printf("tid: %d, batch_idx: %d, row_idx: %d, updated: (%f, %f), recomputed: (%f, %f)\n", thread_idx, batch_idx, row_idx, dA_col_1, dA_col_2, dA_col_r1, dA_col_r2);
  
  if(abs_d1 > E){
    if(!std::isinf(d2)){
      loc = round(d2 / d1) - 1;
      printf("[col check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) update(%f, %f) recompute(%f, %f)\n", (float)d1, (float)d2, loc, dA_col_1, dA_col_2, dA_col_r1, dA_col_r2);
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
    return;
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
    return;
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