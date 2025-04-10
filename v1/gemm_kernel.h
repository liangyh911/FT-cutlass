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

/////////////////////////////////////////////////////////////////////////////////////////////////

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

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, 
                  uint8_t *Signature_Array, uint8_t *Tile_Offset_m, uint8_t *Tile_Offset_n, int *Lock_Signature) {

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
    unsigned int nsm;
    asm volatile("mov.u32  %0, %nsmid;" : "=r"(nsm));
    // printf("Num of SM: %d\n", nsm);
    
    // get thread id in each SM
    // unsigned int tid;
    // asm volatile("mov.u32 %0,%tid.x;" : "=r"(tid));
    // printf("SM id: %d, tid: %d, thread_idx;%d\n", smid, tid, thread_idx);
    
    // int off_A = tb_offset_A.row() + tb_offset_A.column()*params.problem_size.m();
    // int off_B = tb_offset_B.row() + tb_offset_B.column()*params.problem_size.n();

    int off_A = tb_offset_A.column() + tb_offset_A.row()*params.problem_size.k();
    int off_B = tb_offset_B.column() + tb_offset_B.row()*params.problem_size.n();
    
    // printf("M: %d, N: %d, K: %d \n", params.problem_size.m(), problem_size_k, params.problem_size.n());

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

    // int off_C = threadblock_offset.row() + threadblock_offset.column() * params.problem_size.m();
    int off_C = threadblock_offset.column() + threadblock_offset.row() * params.problem_size.n();

    // printf("SM id: %d, Block idx: %d, A_data: %f, A_row_offset: %d, A_col_offset: %d, B_data: %f, B_row_offset: %d, B_col_offset: %d, D_data: %f, D_row_offset: %d, D_col_offset: %d\n", 
    //           smid, block_idx, *(params.ref_A.data()+off_A), tb_offset_A.row(), tb_offset_A.column(), 
    //                           *(params.ref_B.data()+off_B), tb_offset_B.row(), tb_offset_B.column(),
    //                           *(params.ref_D.data()+off_C), threadblock_offset.row(), threadblock_offset.column());

    // if (smid == 0 && thread_idx==0){
    //   for(int r=0; r<problem_size.m(); r++){
    //     for(int c=0; c<problem_size.n(); c++){
    //       int idx = 
    //     }
    //   }
    // }

    // printf("A: idx_0: %f, idx_1: %f, idx_2: %f, idx_3: %f \n", 
    //         *(params.ref_A.data()), *(params.ref_A.data()+1), *(params.ref_A.data()+2), *(params.ref_A.data()+3));
    
    // printf("B: idx_0: %f, idx_1: %f, idx_2: %f, idx_3: %f \n", 
    //         *(params.ref_B.data()), *(params.ref_B.data()+1), *(params.ref_B.data()+2), *(params.ref_B.data()+3));

    
    // 
    // Signature and Find (Matrix SM) (Column Checksum Only)
    //

    // __shared__ unsigned int next_chk_smid, next_matrix_smid;
    __shared__ int next_matrix_block_idx, next_chk_block_idx, flag;

    int tmp_matrix_blk, tmp_chk_blk, tmp_flag;
    // block view
    if(thread_idx == 0){
      if (threadblock_tile_offset.m() != (params.grid_tiled_shape.m() - 1)){
        // Signature for Matrxi SM
        *(Signature_Array + block_idx) = (uint8_t)smid;
        // *(Tile_Offset_m + block_idx) = (uint8_t)threadblock_tile_offset.m();
        // *(Tile_Offset_n + block_idx) = (uint8_t)threadblock_tile_offset.n();
      
        // Find Finished (Naive: get next)
        unsigned int next_matrix_smid, next_chk_smid;
        uint8_t matrix_block_idx = block_idx;
        uint8_t chk_block_idx;
        // printf("SM id:%d, SM array: %d, 1st next: %d\n", smid, *(Signature_Array + smid), next_matrix_smid);
        // __syncthreads();
        bool need_lock = true;
        while (need_lock) {
          matrix_block_idx = (matrix_block_idx + 1) % (params.grid_tiled_shape.m() * params.grid_tiled_shape.n());
          // lock for matrix SM selection
          if (atomicCAS((Lock_Signature + matrix_block_idx), 0, 1) == 0) {
            // get the corresponding chksum SM blk index
            int n = (matrix_block_idx + 1) / params.grid_tiled_shape.m();
            chk_block_idx = params.grid_tiled_shape.m() * (n + 1) - 1;
            // lock for the chksum SM
            if (atomicCAS((Lock_Signature + chk_block_idx), 0, 1) == 0) {
              if ((matrix_block_idx + 1) % params.grid_tiled_shape.m() != 0 &&
                  *(Signature_Array + matrix_block_idx) != 255 && 
                  *(Signature_Array + chk_block_idx) != 255) {
                
                next_matrix_smid = *(Signature_Array + matrix_block_idx);
                next_chk_smid = *(Signature_Array + chk_block_idx);

                tmp_matrix_blk = matrix_block_idx;
                tmp_chk_blk = chk_block_idx;

                *(Signature_Array + matrix_block_idx) = 255;
                need_lock = false;
              }
              // Release the lock
              atomicExch((Lock_Signature + chk_block_idx), 0);
              // printf("current SM: %d, next SM: %d\n", smid, next_matrix_smid);
            }
            atomicExch((Lock_Signature + matrix_block_idx), 0);
          }
        }

        // Check chksum smid == matrix smid
        if(next_chk_smid == next_matrix_smid){
          tmp_flag = 0;
          // printf("Recompute chksum using current SM\n");
        }
        // Check chksum smid == current smid
        else if (smid == next_chk_smid){
          tmp_flag = 1;
          // printf("Current SM is the same as chksum SM\n");
        }
        // SM ids are not the same
        else{
          tmp_flag = 2;
          // printf("Check\n");
          printf("Check. block idx: %d, tile_offset.m: %d, title_offset.n: %d, current SM: %d, next matrix SM: (%d, %d), next chk SM: (%d, %d)\n", 
                  block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), smid, next_matrix_smid, tmp_matrix_blk, next_chk_smid, tmp_chk_blk);
        }
      }
      else{
        // Signature for Checksum (encoded) SM
        *(Signature_Array + block_idx) = (uint8_t)smid;
        // *(Tile_Offset_m + (nsm + smid)) = (uint8_t)threadblock_tile_offset.m();
        // *(Tile_Offset_n + (nsm + smid)) = (uint8_t)threadblock_tile_offset.n();

        // printf("chksum. block_idx: %d, tile_offset.m: %d, title_offset.n: %d, SM: %d, \n", 
        //         block_idx, threadblock_tile_offset.m(), threadblock_tile_offset.n(), *(Signature_Array + block_idx));
      }

      next_matrix_block_idx = tmp_matrix_blk;
      next_chk_block_idx = tmp_chk_blk;
      flag = tmp_flag;
    }
    __syncthreads();

    // begin chkeck
    if(flag == 2){
      int ColBlkOffset = ((next_matrix_block_idx + 1) / params.grid_tiled_shape.m());
      int RowBlkOffset = ((next_matrix_block_idx + 1) % params.grid_tiled_shape.m() - 1);

      int matrix_start_idx = (ColBlkOffset * 128) + (RowBlkOffset * 128) * params.problem_size.n() + thread_idx;
      int chk_start_idx = (ColBlkOffset * 2) + (RowBlkOffset * 128) * params.problem_size.n() + thread_idx;
      float sum = 0;
      uint8_t diff = 0;

      for(int r = 0; r < 128; r++){
        int idx = matrix_start_idx + r * params.problem_size.n();
        sum += *(params.ref_D.data() + idx);
      }

      if(sum == *(params.ref_D.data() + chk_start_idx)){
        diff = 1;
        printf("Not Detect Error at current SM: %d\n", smid);
      }
      else{
        printf("current: %f, next: %f\n", sum, *(params.ref_D.data() + chk_start_idx));
      }

      // if(thread_idx == 0){
      //   printf("current SM: %d, selected blk: %d, Offset: (%d, %d)\n", next_matrix_block_idx, RowOffset, ColOffset);
      //   //  printf("current id: (%d, %d), selected blk: (%d, %d)\n", smid, thread_idx, next_matrix_block_idx, next_chk_block_idx);
      // }
    }
    
    // if(thread_idx == 1){
    //   printf("SM: %d, next SM: %d, next chk SM: %d \n", smid, next_matrix_smid, next_chk_smid);
    // }

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

