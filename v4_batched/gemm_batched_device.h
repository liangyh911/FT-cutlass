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
    \brief Template for a pipelined batch GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_batched.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#define checkCudaErrors(val) check ( (val), #val, __FILE__, __LINE__ )
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

/*! Gemm device-level operator. This is an interface to efficient CUTLASS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:
    
    1. At compile time, it maps data types and high-level structural parameters onto 
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect 
  most configurations to be specified at this level. Applications with more exotic requirements 
  may construct their kernels of interest using CUTLASS components at the threadblock, warp, 
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS GEMM operator implementing the functionality of cuBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the CUTLASS GEMM operator.
    //

    cutlass::gemm::device::Gemm<
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    cutlass::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_A,
      {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_B,
      {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
      {D, ldd},                           // TensorRef<float, layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementA,
      
      /// Layout type for A matrix operand
      typename LayoutA,
      
      /// Element type for B matrix operand
      typename ElementB,
      
      /// Layout type for B matrix operand
      typename LayoutB,
      
      /// Element type for C and D matrix operands
      typename ElementC,
      
      /// Layout type for C and D matrix operands
      typename LayoutC,
      
      /// Element type for internal accumulation
      typename ElementAccumulator,

      /// Operator class tag
      typename OperatorClass,
      
      /// Tag indicating architecture to tune for.  This is the minimum SM that
      /// supports the intended feature. The device kernel can be built
      /// targeting any SM larger than this number.
      typename ArchTag,
      
      /// Threadblock-level tile size (concept: GemmShape)
      typename ThreadblockShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename WarpShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename InstructionShape,
      
      /// Epilogue output operator
      typename EpilogueOutputOp,
      
      /// Threadblock-level swizzling operator
      typename ThreadblockSwizzle,
      
      /// Number of stages used in the pipelined mainloop
      int Stages
    >
    class Gemm;
*/

__device__ int *SM_check_res;

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmBatchedIdentityThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator
>
class GemmBatched {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  using Operator = Operator_;

  /// Define the kernel
  using DefaultGemmKernel = typename kernel::DefaultGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    false,
    Operator
  >::GemmKernel;

  using GemmKernel = kernel::GemmBatched<typename DefaultGemmKernel::Mma, typename DefaultGemmKernel::Epilogue, ThreadblockSwizzle>;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    int64_t stride_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    int64_t stride_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    int64_t stride_C;
    TensorRef<ElementC, LayoutC> ref_D;
    int64_t stride_D;
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      int64_t stride_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      int64_t stride_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      int64_t stride_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      int64_t stride_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int batch_count_
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      stride_A(stride_A_),
      ref_B(ref_B_),
      stride_B(stride_B_),
      ref_C(ref_C_),
      stride_C(stride_C_),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_) { }
  };

private:

  /// Kernel parameters object
  typename GemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  GemmBatched() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!TensorRef_aligned(args.ref_A, kAlignmentA) || (args.stride_A % kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_B, kAlignmentB) || (args.stride_B % kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_C, kAlignmentC) || (args.stride_C % kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_D, kAlignmentC) || (args.stride_D % kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    return 0;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.batch_count);

    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.stride_A,
      args.ref_B.non_const_ref(),
      args.stride_B,
      args.ref_C.non_const_ref(),
      args.stride_C,
      args.ref_D,
      args.stride_D,
      args.epilogue,
      args.batch_count
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    params_.ref_A.reset(args.ref_A.non_const_ref().data());
    params_.ref_B.reset(args.ref_B.non_const_ref().data());
    params_.ref_C.reset(args.ref_C.non_const_ref().data());
    params_.ref_D.reset(args.ref_D.data()); 

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(int if_split_phase, int partion, cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    dim3 grid_gemm(132,1,1);

    dim3 grid_updatechk(132,1,1);
    dim3 block_updatechk(1024,1, 1);
    
    cudaStream_t stream_colchk;
    // cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_colchk);

    bool deBug = true;
    int iterations = 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t_gemm = 0, t_chksum = 0, t_check = 0;

    cudaMalloc((void**)&SM_check_res, 132 * sizeof(int));
    cudaMemset(SM_check_res, 0, 132 * sizeof(int));

    // printf("Grdi: (%d, %d, %d); Blocks: (%d, %d, %d)\n", new_grid.x, new_grid.y, new_grid.z, block.x, block.y, block.z);

    // printf("(%d, %d, %d), %d\n", grid.x, grid.y, grid.z, block.x);

    cudaError_t result;

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    printf("share memory size: %d\n", smem_size);
    
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        printf("error\n");
        return Status::kErrorInternal;
      }
    }

    int batch_per_TB = (int)(ceil((double)block_updatechk.x / (double)params_.problem_size.n()));
    // if(batch_per_TB > 6) batch_per_TB = 6;
    int B = (batch_per_TB > 6) ? 6 : batch_per_TB;
    int update_smem_size = B * 2 * params_.problem_size.k() * sizeof(float);

    // 128 96 120
    int matrix_SM = (if_split_phase == 2)? 132 : 128;
    
    void *kernelArgs[] = {&params_, &if_split_phase, &SM_check_res, &partion, &matrix_SM};

    cutlass::arch::synclog_setup();

    if(if_split_phase == 0 || if_split_phase == 1) {
      cutlass::update_checksum<GemmKernel><<<grid_updatechk, block_updatechk, update_smem_size, stream_colchk>>>(params_, matrix_SM, batch_per_TB);
      // cutlass::update_checksum_v2<GemmKernel><<<grid_updatechk, block_updatechk, update_smem_size, stream_colchk>>>(params_, matrix_SM, batch_per_TB);
      // cutlass::update_checksum_v3<GemmKernel><<<grid_updatechk, block_updatechk, update_smem_size, stream_colchk>>>(params_, matrix_SM, batch_per_TB);
    }
    cudaLaunchCooperativeKernel((void*)cutlass::Kernel<GemmKernel>, grid_gemm, block, kernelArgs, smem_size, stream);
    // cutlass::Kernel<GemmKernel><<<grid_gemm, block, smem_size, stream>>>(params_, if_split_phase, SM_check_res, partion, matrix_SM);
    if(if_split_phase == 0) cutlass::check_SM<GemmKernel><<<grid_gemm, block_updatechk, 0, stream>>>(params_, matrix_SM, SM_check_res);

    cudaDeviceSynchronize();
    // if(deBug){
    //   cudaEventRecord(start, stream);
    // }

    float sum_gemm = 0, sum_chksum = 0.f, sum_check = 0.f;

    for(int i = 0; i < iterations; i++){

      if(deBug && (if_split_phase == 0 || if_split_phase == 1)){
        cudaEventRecord(start, stream_colchk);
      }
      if(if_split_phase == 0 || if_split_phase == 1) {
        cutlass::update_checksum<GemmKernel><<<grid_updatechk, block_updatechk, update_smem_size, stream_colchk>>>(params_, matrix_SM, batch_per_TB);
        // cutlass::update_checksum_v2<GemmKernel><<<grid_updatechk, block_updatechk, update_smem_size, stream_colchk>>>(params_, matrix_SM, batch_per_TB);
        // cutlass::update_checksum_v3<GemmKernel><<<grid_updatechk, block_updatechk, update_smem_size, stream_colchk>>>(params_, matrix_SM, batch_per_TB);
      }
      if(deBug && (if_split_phase == 0 || if_split_phase == 1)){
        cudaEventRecord(stop, stream_colchk);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_chksum, start, stop);

        sum_chksum += t_chksum;
      }

      if(deBug){
        cudaEventRecord(start, stream);
      }   
      cudaLaunchCooperativeKernel((void*)cutlass::Kernel<GemmKernel>, grid_gemm, block, kernelArgs, smem_size, stream);
      // cutlass::Kernel<GemmKernel><<<grid_gemm, block, smem_size, stream_main>>>(params_, if_split_phase, SM_check_res, partion);
      // if(if_split_phase == 0) cutlass::check_SM<GemmKernel><<<grid_gemm, block_updatechk, 0, stream>>>(params_, matrix_SM, SM_check_res);
      if(deBug){
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_gemm, start, stop);
        sum_gemm += t_gemm;
      }

      if(deBug && if_split_phase == 0){
        cudaEventRecord(start, stream);
      }  
      if(if_split_phase == 0) cutlass::check_SM<GemmKernel><<<grid_gemm, block_updatechk, 0, stream>>>(params_, matrix_SM, SM_check_res);
      if(deBug && if_split_phase == 0){
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_check, start, stop);
        sum_check += t_check;
      }

      cudaDeviceSynchronize();
    }

    printf("gemm kernel time: %f, update kernel time: %f, check phase: %f \n", sum_gemm/iterations, sum_chksum/iterations, sum_check/iterations);

    cudaFree(SM_check_res);

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for column-major output exchanges problem size and operand.
template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Operator class tag
  typename OperatorClass_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape_,
  /// Epilogue output operator
  typename EpilogueOutputOp_,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle_,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Access granularity of A matrix in units of elements
  int AlignmentA,
  /// Access granularity of B matrix in units of elements
  int AlignmentB,
  typename Operator_
>
class GemmBatched<
  ElementA_,
  LayoutA_,
  ElementB_,
  LayoutB_,
  ElementC_,
  layout::ColumnMajor,
  ElementAccumulator_,
  OperatorClass_,
  ArchTag_,
  ThreadblockShape_,
  WarpShape_,
  InstructionShape_,
  EpilogueOutputOp_,
  ThreadblockSwizzle_,
  Stages,
  AlignmentA,
  AlignmentB,
  Operator_
> {
public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = layout::ColumnMajor;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;

  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = false;

  //
  using UnderlyingOperator = GemmBatched< 
    ElementB,
    typename layout::LayoutTranspose<LayoutB>::type,
    ElementA,
    typename layout::LayoutTranspose<LayoutA>::type,
    ElementC,
    layout::RowMajor,    
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    kAlignmentB,
    kAlignmentA
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using GemmKernel = typename UnderlyingOperator::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    int64_t stride_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    int64_t stride_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    int64_t stride_C;
    TensorRef<ElementC, LayoutC> ref_D;
    int64_t stride_D;
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      int64_t stride_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      int64_t stride_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      int64_t stride_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      int64_t stride_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int batch_count_
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      stride_A(stride_A_),
      ref_B(ref_B_),
      stride_B(stride_B_),
      ref_C(ref_C_),
      stride_C(stride_C_),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_) { }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  GemmBatched() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {
    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      args.stride_B,
      {args.ref_A.data(), args.ref_A.stride(0)},
      args.stride_A,
      {args.ref_C.data(), args.ref_C.stride(0)},
      args.stride_C,
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.stride_D,
      args.epilogue,
      args.batch_count
    );
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace);
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  Status run(int if_split_phase, int partion, cudaStream_t stream = nullptr) {
    return underlying_operator_.run(if_split_phase, partion, stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    int if_split_phase, int partion, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(if_split_phase, partion, stream);
    }

    return status;
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
