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
  GemmBatched() = default;

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // get SM id
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    int threadblock_tile_offset_m, threadblock_tile_offset_k, threadblock_tile_offset_n;

    // SM based schudule
    // assign enough SMs for each batch 
    int SM_per_batch = params.grid_tiled_shape.m() * params.grid_tiled_shape.n();
    int batch_step = (int)(floor((double)132 / (double)SM_per_batch));
    int local_smid = smid % SM_per_batch;
    int init_batch_idx = smid / SM_per_batch;
    if(init_batch_idx >= batch_step){
      return;
    }
    smid = local_smid;

    // 2nd split SM for each matrix
    int checksumblk_per_col = 0;
    int matrix_shape_m = params.grid_tiled_shape.m() - checksumblk_per_col;

    int max_col = (int)ceil((double)SM_per_batch / (double)(matrix_shape_m));
    if(max_col > params.grid_tiled_shape.n()){
      max_col = params.grid_tiled_shape.n();
    }

    int remaining_SM = (int)(max_col * checksumblk_per_col);
    int matrix_SM = (int)(SM_per_batch - remaining_SM);

    int matrix_next_blk_offset_m = matrix_SM % (matrix_shape_m);
    int matrix_next_blk_offset_n = (matrix_SM / matrix_shape_m);
    int checksum_next_blk_offset_n = (checksumblk_per_col != 0) ? (remaining_SM / checksumblk_per_col) : 0;
    // iteration based on GeMM not (GeMM + chksum)
    int SM_iter = (int)ceil((double)((matrix_shape_m * params.grid_tiled_shape.n())/(double)matrix_SM));     

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


    // Each CTA handles multiple batch indices to accommodate limited range of CUDA grid's Z dimension
    for(int batch_idx = init_batch_idx; batch_idx < params.batch_count; batch_idx += batch_step) {
      bool beyond_bound = false;
      for(int iter = 0; iter < SM_iter; iter++){  
        if(smid < matrix_SM){
          // for matrix
          int add_col = 0;
          threadblock_tile_offset_m = (smid % (matrix_shape_m) + iter * matrix_next_blk_offset_m) % (matrix_shape_m);
          add_col = ((smid % (matrix_shape_m) + iter * matrix_next_blk_offset_m) / (matrix_shape_m));
          threadblock_tile_offset_n = smid / (matrix_shape_m) + iter * matrix_next_blk_offset_n + add_col;
        }
        else{
          // for checksum          
          unsigned int local_chk_blk_idx = smid - matrix_SM;
          threadblock_tile_offset_m = matrix_shape_m + (local_chk_blk_idx % checksumblk_per_col);
          threadblock_tile_offset_n = local_chk_blk_idx / checksumblk_per_col + iter * checksum_next_blk_offset_n;
        }
        if(threadblock_tile_offset_n >= params.grid_tiled_shape.n()){
          // return;
          beyond_bound = true;
        }
        
        if(!beyond_bound){
          // Compute initial location in logical coordinates
          cutlass::MatrixCoord tb_offset_A{
            threadblock_tile_offset_m * Mma::Shape::kM,
            0
          };

          cutlass::MatrixCoord tb_offset_B{
            0,
            threadblock_tile_offset_n * Mma::Shape::kN
          };

          // Compute position within threadblock
          int thread_idx = threadIdx.x;

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
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

