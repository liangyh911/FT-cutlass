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
#include <math.h>

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

  __device__ void update_col_v3(Params const &params, int thread_idx, int batch_idx){

    int M = params.problem_size.m();
    int K = params.problem_size.k();
    int N = params.problem_size.n();
    int tiled_N = blockDim.x; 

    int iter = (int)(ceil((double)N / (double)tiled_N));

    for(int i = 0; i < iter; i++){
      int col_idx = (i * tiled_N) + thread_idx;
      if(col_idx < N){
        for(int m = 0; m < 2; m++){
          float accum = 0.f;
          int idx_a = (batch_idx * params.stride_A) + ((M + m) * K);
          int idx_b = (batch_idx * params.stride_B) + col_idx;
          int idx_chk = (batch_idx * params.stride_D) + (M + m) * N + col_idx;
        
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

  // __device__ void check_phase_v2(Params const &params, int batch_idx, int thread_idx, int col_idx, int *SM_check_res, int matrix_SM, int batch_step, int &diff, int &loc){
    
  //   int M = params.problem_size.m();
  //   int K = params.problem_size.k();
  //   int N = params.problem_size.n();
  //   float E = 100;
  //   // int loc = -1;
  //   float MAX = 0;
  //   // int diff = 0;

  //   // recompute checksum (no weighted, weighted)
  //   float dA_col_r1 = 0.f;
  //   float dA_col_r2 = 0.f;
    
  //   int start_idx = (params.stride_D * batch_idx) + col_idx;
    
  //   #pragma unroll 128
  //   for(int r = 0; r < M; r++){
  //     int idx = start_idx + r * N;
  //     float element = (float)*(params.ref_D.data() + idx);
      
  //     dA_col_r1 += element;
  //     dA_col_r2 += (float)(r+1) * element;
  //   }

  //   // detect error
  //   float dA_col_1 = *(params.ref_D.data() + start_idx + (M*N));
  //   float dA_col_2 = *(params.ref_D.data() + start_idx + (M+1)*N);

  //   float d1 = (float)(dA_col_1 - dA_col_r1);
  //   float d2 = (float)(dA_col_2 - dA_col_r2);
  //   float abs_d1 = fabs(d1);

  //   // printf("tid: %d, batch_idx: %d, row_idx: %d, updated: (%f, %f), recomputed: (%f, %f)\n", thread_idx, batch_idx, row_idx, dA_col_1, dA_col_2, dA_col_r1, dA_col_r2);
    
  //   if(abs_d1 > E){
  //     if(!std::isinf(d2)){
  //       loc = round(d2 / d1) - 1;
  //       printf("[col check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) update(%f, %f) recompute(%f, %f)\n", (float)d1, (float)d2, loc, dA_col_1, dA_col_2, dA_col_r1, dA_col_r2);
  //       diff = 1;
  //     }
  //     else{
  //       MAX = 0;
	// 			int counter = 0;
	// 			for(int i = 0; i < N; i++) {
	// 				if(fabs((float)*(params.ref_D.data() + start_idx + i * N)) > MAX){
	// 					MAX = fabs((float)*(params.ref_D.data() + start_idx + i * N));
	// 					loc = i;
	// 				}
	// 				if(fabs((float)*(params.ref_D.data() + start_idx + i * N)) > 1e10){
	// 					counter++;
	// 					if(counter > 1){
	// 						printf("[col check]col chksum error, more than one large number. (d1 = %.6f, d2 = %.6f)\n",(float)d1, (float)d2);
	// 						return;
	// 					}
	// 				}
	// 			}
	// 			printf("[col check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
  //       diff = 1;        
  //     }
  //     return;
  //   }
  //   // abs == inf
  //   if(std::isinf(abs_d1)){
  //     MAX = 0;
  //     int64_t counter = 0;
  //     for(int i = 0; i < N; i++) {
  //       if(fabs((float)*(params.ref_D.data() + start_idx + i * N)) > MAX){
  //         MAX = fabs((float)*(params.ref_D.data() + start_idx + i * N));
  //         loc = i;
  //       }
  //       if(std::isinf(*(params.ref_D.data() + start_idx + i * N)) || fabs((float)*(params.ref_D.data() + start_idx + i * N)) > 1e10){
  //         counter++;
  //         if(counter > 1){
  //           printf("[col check]Multi INFs or Large Number detected in one column.(d1 = %.6f, d2 = %.6f, iter = %d)\n", (float)d1, (float)d2, i);
  //           return;
  //         }
  //       }
  //     }
  //     if(counter == 0){
  //       printf("[col chk]No INF or Large Number found.\n");
  //       return;
  //     }
  //     printf("[col check]INF detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
  //     diff = 1;
  //   }
  //   // abs == nan
	//   if(std::isnan(abs_d1)){
  //     int64_t counter = 0;
  //     for(int i = 0; i < N; i++) {
  //       if (std::isnan(*(params.ref_D.data() + start_idx + i * N))) {
  //         loc = i;
  //         counter++;
  //       }
  //       if(std::isinf(*(params.ref_D.data() + start_idx + i * N))){
  //         counter++;
  //       }
  //       if(fabs((float)*(params.ref_D.data() + start_idx + i * N)) > 1e10){
  //         counter++;
  //       }
  //       if(counter > 1){
  //         printf("[col check]Multi INF, NAN or Large Number detected in one column. (iter = %d)\n", i);
  //         return;
  //       }
  //     }
  //     printf("[col check]NAN detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
  //     diff = 1;
  //   }
    
  //   // if(diff != 0){
  //   //   // Locate corrupted SM
  //   //   int error_n_offset = row_idx / blockDim.x;
  //   //   int error_m_offset = loc / 128;
  //   //   int error_local_smid = error_m_offset + error_n_offset * params.grid_tiled_shape.m();
  //   //   int error_smid = error_local_smid + matrix_SM * batch_idx % batch_step;

  //   //   // record results
  //   //   // Atomic sum
  //   //   atomicAdd((SM_check_res + error_smid), diff);
  //   // }
  //   // __syncthreads();
  // }
  __device__ void check_phase_v2(Params const &params, int batch_idx, int thread_idx, int col_idx, int *SM_check_res, int matrix_SM, int batch_step, int &diff, int &loc){
    
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
    
    // if(diff != 0){
    //   // Locate corrupted SM
    //   int error_n_offset = row_idx / blockDim.x;
    //   int error_m_offset = loc / 128;
    //   int error_local_smid = error_m_offset + error_n_offset * params.grid_tiled_shape.m();
    //   int error_smid = error_local_smid + matrix_SM * batch_idx % batch_step;

    //   // record results
    //   // Atomic sum
    //   atomicAdd((SM_check_res + error_smid), diff);
    // }
    // __syncthreads();
  }


  template <typename T>
  __device__ void force_bit_one_f32(T *dA, int bit, int *count, float *buf){ 
    // 30 or 29
    float orgValue = (float)*(dA);
    float tmp = (float)*(dA);
    // printf("%.4f ", orgValue);
    
    uint32_t* intValue = reinterpret_cast<uint32_t*>(&orgValue);
    *intValue |= (1u << bit);
    // *intValue &= ~ ((1u << bit));
    *(dA) = (T) *reinterpret_cast<float*>(intValue);
    
    if(tmp != *(dA)){
      // printf("%.4f %.4f ", tmp, *(dA));
      // int idx = (*count) * 2;
      int idx = *count;
      *(buf + idx) = tmp;
      *(buf + (idx + 1)) = *(dA);
      (*count) += 2;
    }
    // printf("%.4f ", *(dA));
  }

  template <typename T>
  __device__ void force_bit_one_bf16(T *dA, int bit, int *count, float *buf){ 
    // 30 or 29
    float orgValue = static_cast<float>(*dA);
    float tmp = orgValue;
    // printf("%.4f ", orgValue);
    
    // uint32_t* intValue = reinterpret_cast<uint32_t*>(&orgValue);
    uint32_t intValue = *reinterpret_cast<uint32_t*>(&orgValue);
    uint16_t bf16_bits = static_cast<uint16_t>(intValue >> 16);
    bf16_bits |= (1u << bit);
    // *intValue &= ~ ((1u << bit));

    uint32_t new_int_value = (static_cast<uint32_t>(bf16_bits) << 16);
    float new_float = *reinterpret_cast<float*>(&new_int_value);
    *dA = static_cast<cutlass::bfloat16_t>(new_float);

    if(tmp != new_float){
      // printf("%.4f %.4f ", tmp, new_float);
      // int idx = (*count) * 2;
      int idx = *count;
      *(buf + idx) = tmp;
      *(buf + (idx + 1)) = new_float;
      (*count) += 2;
    }
    // printf("%.4f ", *(dA));
  }

  GemmBatched() = default;

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, 
                    int if_split_phase, int *SM_check_res, int nSM, int monitored_batched_count,
                    int faulty_smid, int *faulty_MMAs, int *faulty_elements, int faulty_bit, int *counter, float *buf) {

    // get SM id
    unsigned int real_smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(real_smid));
    int n_smid;
    asm volatile("mov.u32 %0, %nsmid;" : "=r"(n_smid));
    int threadblock_tile_offset_m, threadblock_tile_offset_k, threadblock_tile_offset_n;
    
    // int nSM = 128;
    // return update checksum SM
    if(real_smid > (nSM - 1)) return;
    
    // if(threadIdx.x == 0) {
    //   printf("gemm smid: %d\n", real_smid);
    // }

    // SM based schudule
    // assign enough SMs for each batch 
    int SM_per_batch = params.grid_tiled_shape.m() * params.grid_tiled_shape.n();
    if(SM_per_batch > nSM){
      SM_per_batch = nSM;
    }
    int batch_step = (int)(floor((double)nSM / (double)SM_per_batch));
    int local_smid = real_smid % SM_per_batch;
    int init_batch_idx = real_smid / SM_per_batch;
    int batch_iter = (int)(ceil((double)params.batch_count / (double)batch_step));

    int smid = local_smid;

    // if(threadIdx.x == 0) printf("SM_per_batch: %d, batch_step: %d, batch_iter: %d\n", SM_per_batch, batch_step, batch_iter);
    // if(threadIdx.x == 0) printf("matrix_SM: %d, M: %d, N: %d\n", matrix_SM, params.problem_size.m(), params.problem_size.n());
    
    int checksumblk_per_col = 0;
    // int matrix_SM = SM_per_batch;
    int matrix_shape_m = params.grid_tiled_shape.m();

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

    // Each CTA handles multiple batch indices to accommodate limited range of CUDA grid's Z dimension    
    int batch_idx;
    // if(real_smid < (nSM - 1)){
      // for matrix
      for(int b_iter = 0; b_iter < batch_iter; b_iter += 1) {
        batch_idx = init_batch_idx + b_iter * batch_step;
        int local_matrix_idx = smid;
        block_idx = local_matrix_idx + (local_matrix_idx / matrix_shape_m) * checksumblk_per_col;
        block_to_coordinate(block_idx, params.grid_tiled_shape.m(), threadblock_tile_offset_m, threadblock_tile_offset_n);

        if(batch_idx < params.batch_count){
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

        // Simulate compute matrix error
        // if(batch_idx == 1){
        //   if(thread_idx == 0){
        //     int inject_idx = batch_idx * params.stride_D;

        //     *(params.ref_D.data() + inject_idx) = 0;
        //   }
        //   // __syncthreads();
        // }
        
        // Fault Injection
        if(real_smid == faulty_smid && thread_idx == 0){
          // int mma_grid_m = params.problem_size.m() / 16;
          // int mma_grid_n = params.problem_size.n() / 8;
          int N = params.problem_size.n();
          int c = 0;
          for(int i = 0; i < 64; i++){
            int mma_m = (threadblock_tile_offset_m * 128) + (faulty_MMAs[i] % 8) * 16;
            int mma_n = (threadblock_tile_offset_n * 256) + (faulty_MMAs[i] / 8) * 8;

            // index of 1st faulty element
            int fault_m = faulty_elements[i] % 8;
            int fault_n = faulty_elements[i] / 8;
            int idx = (mma_m + fault_m) * N + (mma_n + fault_n);
            force_bit_one_bf16((params.ref_D.data()+idx), faulty_bit, counter, buf);

            // index of 2nd faulty element (gap is 64)
            fault_m += 8;
            idx = (mma_m + fault_m) * N + (mma_n + fault_n);
            force_bit_one_bf16((params.ref_D.data()+idx), faulty_bit, counter, buf);
          }
        }
        __syncthreads();

        // if(real_smid == faulty_smid && (thread_idx == faulty_tid_1)){
        //   // printf("batched injection. sm: %d, tid1: %d, bit: %d\n", faulty_smid, faulty_tid_1, faulty_tid_2, faulty_bit);
        //   int thread_tiled_m = (threadblock_tile_offset_m * 128) + ((thread_idx % 8) * 16);
        //   int thread_tiled_n = (threadblock_tile_offset_n * 256) + ((thread_idx / 8) * 8);

        //   // int M = (if_split_phase == 0) ? (params.problem_size.m()+2) : params.problem_size.m();
        //   int N = params.problem_size.n();
        //   // int bit = 20;
          
        //   // printf("[ \n");
        //   for(int i = thread_tiled_m; i < (thread_tiled_m+16); i++){
        //     for(int j = thread_tiled_n; j < (thread_tiled_n+8); j++){
        //       int idx = j + i * N + batch_idx * params.stride_D;
        //       // force_bit_one_f32((params.ref_D.data()+idx), faulty_bit, counter, buf);
        //       force_bit_one_bf16((params.ref_D.data()+idx), faulty_bit, counter, buf);
        //     }
        //     // printf("\n");  
        //   }
        //   // printf("] \n");
        // }
        // __syncthreads();

        // if(real_smid == faulty_smid && (thread_idx == faulty_tid_2)){
        //   // printf("batched injection. sm: %d, tid1: %d, bit: %d\n", faulty_smid, faulty_tid_1, faulty_tid_2, faulty_bit);
          
        //   int thread_tiled_m = (threadblock_tile_offset_m * 128) + ((thread_idx % 8) * 16);
        //   int thread_tiled_n = (threadblock_tile_offset_n * 256) + ((thread_idx / 8) * 8);

        //   // int M = (if_split_phase == 0) ? (params.problem_size.m()+2) : params.problem_size.m();
        //   int N = params.problem_size.n();

        //   int init_buf_idx = 16*8*2*batch_iter;
        //   // int init_buf_idx = *counter;
        //   // int bit = 20;
          
        //   // printf("[ \n");
        //   for(int i = thread_tiled_m; i < (thread_tiled_m+16); i++){
        //     for(int j = thread_tiled_n; j < (thread_tiled_n+8); j++){
        //       int idx = j + i * N + batch_idx * params.stride_D;
        //       // force_bit_one_f32((params.ref_D.data()+idx), faulty_bit, (counter + 1), (buf + init_buf_idx));
        //       force_bit_one_bf16((params.ref_D.data()+idx), faulty_bit, (counter + 1), (buf + init_buf_idx));
        //     }
        //     // printf("\n");  
        //   }
          
        //   // printf("] \n");
        // }
        // __syncthreads();
      }
    // }

    #if 1
    if(if_split_phase == 0){
      // check checksum
      // using Dtype = typename decltype(params.ref_D)::Element;
      // cooperative_groups::this_grid().sync();
      // if(real_smid < (matrix_SM * batch_step)){
        int check_req_SM = params.grid_tiled_shape.n();
        int check_step = ((int)(floor((double)SM_per_batch / (double)check_req_SM))) * batch_step;
        int check_iter = (int)(ceil((double)monitored_batched_count / (double)check_step));
        int checked_init_batch_idx = ((init_batch_idx + 1) % batch_step) + (smid / check_req_SM) * batch_step;
        
        int last_iter_batch = monitored_batched_count % batch_step;
        
        for(int i = 0; i < check_iter; i += 1){
          if((last_iter_batch != 0) && (i == check_iter - 1)){            
            if (init_batch_idx > last_iter_batch){
              return;
            }
            else if(checked_init_batch_idx == last_iter_batch){
              checked_init_batch_idx = (smid / check_req_SM) * batch_step;
              // checked_init_batch_idx = ((init_batch_idx + 1) % last_iter_batch) + (smid / check_req_SM) * last_iter_batch;
            }
          }
          int checked_batch_idx = checked_init_batch_idx + i * check_step;

          // if(threadIdx.x == 0 && i == check_iter - 1) printf("iter: %d, real smid: %d, local smid: %d, check_iter: %d, init_batch_idx: %d, checked_init_batch_idx: %d, checked_batch_idx: %d\n", i, real_smid, local_smid, check_iter, init_batch_idx, checked_init_batch_idx, checked_batch_idx);
        
          if(checked_batch_idx < monitored_batched_count){
            // if(threadIdx.x == 0) printf("iter: %d, real smid: %d, local smid: %d, check_iter: %d, init_batch_idx: %d, checked_init_batch_idx: %d, checked_batch_idx: %d\n", i, real_smid, local_smid, check_iter, init_batch_idx, checked_init_batch_idx, checked_batch_idx);

            int diff = 0, loc = -1;
    
            // int check_offset = 1;
            // int target_smid = (smid + check_offset) % SM_per_batch;
            // int col_idx = thread_idx + (target_smid / params.grid_tiled_shape.m()) * blockDim.x;
            // int row_idx = (target_smid % params.grid_tiled_shape.m()) * 128;

            int col_idx = thread_idx + (smid % check_req_SM) * blockDim.x;
            // int col_idx = thread_idx + (smid) * blockDim.x;
            
            if(col_idx < params.problem_size.n()){
              check_phase_v2(params, checked_batch_idx, thread_idx, col_idx, SM_check_res, SM_per_batch, batch_step, diff, loc);
              // global in-batch idx
              // int global_idx = row_idx * params.problem_size.n() + col_idx;
              // check_phase_v2(params, checked_batch_idx, thread_idx, global_idx, SM_check_res, SM_per_batch, batch_step, diff, loc);
              
              if(diff != 0){
                // Locate corrupted SM
                int error_n_offset = col_idx / 256;
                int error_m_offset = loc / 128;
                int error_local_smid = error_m_offset + error_n_offset * params.grid_tiled_shape.m();
                int error_smid = error_local_smid + ((init_batch_idx + 1) % batch_step) * SM_per_batch;
      
                // printf("%d Error detected at SM %d by checker SM %d (%d)\n", i, error_smid, real_smid, checked_batch_idx);
                int checksum_SM = n_smid - nSM;
                int chksum_iter = checked_batch_idx / checksum_SM;
                int checksum_smid = ((checked_batch_idx % checksum_SM) + (chksum_iter % checksum_SM)) + nSM;
                
                // int checksum_smid = (checked_batch_idx % (n_smid - nSM)) + nSM;
                printf("%d Error detected at SM %d by checker SM %d. Checksum SM %d (%d)\n", i, error_smid, real_smid, checksum_smid, checked_batch_idx);

                // record results
                // Atomic sum
                atomicAdd((SM_check_res + real_smid), diff);
                atomicAdd((SM_check_res + checksum_smid), diff);
                if (error_smid < n_smid && error_smid >-1) atomicAdd((SM_check_res + error_smid), diff);
              }
              __syncthreads();
            }
          }
        }
      // }
    }
    #endif
    // if(threadIdx.x == 0) printf("init_batch: %d\n", temp);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass