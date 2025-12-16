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
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/layout/permute.h"

// #include "cutlass/gemm_ring_queue.h"
#include <cmath>

#include <string>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

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

// __device__ uint8_t *Signature_Array;
// __device__ int *Lock_Signature;
// __device__ RingQueue_v2 *d_queues;
// __device__ int *d_buffer, *d_head, *d_tail;

__device__ int *SM_check_res_1;
// __device__ int *d_all_start, *d_compute, *d_finding, * d_recompute, *d_compare, *d_checking, *d_SM_JOBS, *d_all_start_for_split;
// __device__ uint8_t *ChkSum_Signature_A_Col;

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
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
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
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute>
class Gemm {
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
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Define the kernel
  using GemmKernel = typename kernel::DefaultGemm<
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
    kSplitKSerial,
    Operator,
    SharedMemoryClearOption::kNone,
    GatherA,
    GatherB,
    ScatterD,
    PermuteDLayout
  >::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ = 
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      int const *gather_A_indices_ = nullptr,
      int const *gather_B_indices_ = nullptr,
      int const *scatter_D_indices_ = nullptr
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      gather_A_indices(gather_A_indices_),
      gather_B_indices(gather_B_indices_),
      scatter_D_indices(scatter_D_indices_) {

    }
  };

private:

  /// Kernel parameters object
  typename GemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  Gemm() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = GemmKernel::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    size_t bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);
    
    if (kSplitKSerial && args.split_k_slices > 1) {

      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (kSplitKSerial) {
      if (args.split_k_slices > 1) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }

        size_t bytes = get_workspace_size(args);
      
        cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    }
    else {

      if (args.split_k_slices > 1) {
        return Status::kErrorInvalidProblem;
      }
    }

    // printf("Row: stride A: %d, stride B: %d, stride C: %d\n", args.ref_A.stride(0), args.ref_B.stride(0), args.ref_C.stride(0));

    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    
    if (kSplitKSerial && args.split_k_slices > 1) {  
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    params_.ref_A.reset(args.ref_A.non_const_ref().data());
    params_.ref_B.reset(args.ref_B.non_const_ref().data());
    params_.ref_C.reset(args.ref_C.non_const_ref().data());
    params_.ref_D.reset(args.ref_D.data());
    params_.output_op = args.epilogue;
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  void recordTime(std::string FP, float time, bool DEBUG){
    std::ofstream outFile(FP, std::ios::app);
    if(!outFile){
      std::cerr << "Failed to open the file for appending." << std::endl;
      return;
    }
    outFile << time << std::endl;
    if(DEBUG) printf("Data appended to the file successfully.\n");
  }

  /// Runs the kernel using initialized state.
  Status run(int if_split_phase, int partion, bool DEBUG, cudaStream_t stream = nullptr) {

    // Preparing time
    cudaEvent_t abft_prepare_start, abft_prepare_end;
    if (DEBUG){
      cudaEventCreate(&abft_prepare_start,0);
      cudaEventCreate(&abft_prepare_end,0);
      cudaEventRecord(abft_prepare_start, 0);
    }
    fs::path destinationFile, fullPath;
    float t1;
    const char* homeDir = nullptr;
    homeDir = getenv("HOME");
    fs::path homePath(homeDir);

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    // printf("m: %d, n: %d, k: %d\n", params_.problem_size.m(), params_.problem_size.n(), params_.problem_size.k());

    cudaError_t result;

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel_GEMM<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    // Fault Injection Info
    char flag;
    bool injection = false;
    char *job_id = getenv("SLURM_JOB_ID");

    int gpu_dev = -1;
    cudaGetDevice(&gpu_dev);
    // int faulty_smid = -1, faulty_tid_1 = -1, faulty_tid_2 = -1, faulty_bit = -1;

    int faulty_smid = -1, faulty_bit = -1, *h_faulty_MMAs, *d_faulty_MMAs, *h_faulty_elements, *d_faulty_elements;
    size_t faulty_size = sizeof(int) * 64;

    h_faulty_MMAs = (int*)malloc(faulty_size);
    h_faulty_elements = (int*)malloc(faulty_size);
    cudaMalloc((void**)&d_faulty_MMAs, faulty_size);
    cudaMemset(d_faulty_MMAs, -1, faulty_size);
    cudaMalloc((void**)&d_faulty_elements, faulty_size);
    cudaMemset(d_faulty_elements, -1, faulty_size);

    int checksumblk_per_col = 0;
    if(if_split_phase == 0 || if_split_phase == 1){
      // if able ABFT
      // checksumblk_per_col = (int)(ceil((double)((params.grid_tiled_shape.m()) / (double)(128))));
      checksumblk_per_col = (int)(ceil((double)((partion) / (double)(128))));
    }
    int matrix_shape_m = params_.grid_tiled_shape.m() - checksumblk_per_col;
    int max_col = (int)ceil((double)132 / (double)(matrix_shape_m));
    if(max_col > params_.grid_tiled_shape.n()){
      max_col = params_.grid_tiled_shape.n();
    }
    int remaining_SM = (int)(max_col * checksumblk_per_col);
    int matrix_SM = (int)(132 - remaining_SM);
    int SM_iter = (int)ceil((double)((matrix_shape_m * params_.grid_tiled_shape.n())/(double)matrix_SM));

    int *d_counter, *h_counter;
    cudaMalloc((void**)&d_counter, 1 * sizeof(int));
    cudaMemset(d_counter, 0, 1 * sizeof(int));
    h_counter = (int*)malloc(1 * sizeof(int));

    float *h_buf, *d_buf;
    // size_t buf_size = (16*8*2*SM_iter * 2) * sizeof(float);
    size_t buf_size = (64 * 2 * SM_iter * 2) * sizeof(float);
    cudaMalloc((void**)&d_buf, buf_size);
    cudaMemset(d_buf, 0, buf_size);
    h_buf = (float*)malloc(buf_size);

    // destinationFile = "/home/yuhangl/control/FI.txt";
    // fs::path FIInfoPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "fi_info.txt";
    // Absolute Path
    // fs::path FIInfoPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "fi_info.bin";
    // destinationFile = fs::path("/home/yuhangl/control_" + std::string(job_id)) / "FI.txt";
    // Relative Path
    fs::path FIInfoPath = fs::path("./control_" + std::string(job_id) + "/" + std::to_string(gpu_dev)) / "fi_info.bin";
    destinationFile = fs::path("./control_" + std::string(job_id) + "/" + std::to_string(gpu_dev)) / "FI.txt";

    std::ifstream FIFile(destinationFile);
    if(FIFile.is_open()){
      FIFile.get(flag);
      if(flag == 't'){
        injection = true;
        // printf("Perform Fault Injection.\n");
        // read injected SM and thread
        // std::ifstream planFile("/home/yuhangl/control/plan.txt");
        // fs::path planPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "plan.txt";
        // std::ifstream planFile(planPath);
        // if(planFile.is_open()){
        //   if (planFile >> faulty_smid >> faulty_tid_1 >> faulty_tid_2) {
        //       // std::cout << "faulty_smid = " << faulty_smid << ", faulty_tid = " << faulty_tid << std::endl;
        //   }
        // }
        // else{
        //   printf("plan: Cannot open file, using default setting.\n");
        // }
        // planFile.close();

        // read the faulty SM, MMAs, elements (faultySM, faultyMMA1, faultyMMA2,..., faultyMMA16, faultyElement1, faultyElement2,..., faultyElement16)
        // Absolute Path
        // fs::path planPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "plan.txt";
        // Relative Path
        fs::path planPath = fs::path("./control_" + std::string(job_id) + "/" + std::to_string(gpu_dev)) / "plan.txt";
        std::ifstream planFile(planPath);
        if(planFile.is_open()){
          std::string line;
          // while (std::getline(planFile, line)) {
            
          if (!std::getline(planFile, line)) {
              std::cerr << "File is empty" << std::endl;
              return Status::kErrorInternal;
          }

          std::stringstream ss(line);
          std::string token;
          std::vector<int> nums;

          while (std::getline(ss, token, ' ')) {
              nums.push_back(std::stoi(token));
          }

          if (nums.size() != 129) {
              printf("Error: expected 129 numbers but got %ld\n", nums.size());
              return Status::kErrorInternal;
          }

          int idx = 0;
          faulty_smid = nums[idx++];
          // printf("faulty SM: %d, faulty MMA: ", faulty_smid);

          for (int i = 0; i < 64; i++){
            h_faulty_MMAs[i] = nums[idx++];
            // printf("%d ", h_faulty_MMAs[i]);
          }

          // printf("faulty elements: ");
          for (int i = 0; i < 64; i++){
            h_faulty_elements[i] = nums[idx++];
            // printf("%d ", h_faulty_elements[i]);
          }
          // printf("\n");
              
          // }

          cudaMemcpy(d_faulty_MMAs, h_faulty_MMAs, faulty_size, cudaMemcpyHostToDevice);
          cudaMemcpy(d_faulty_elements, h_faulty_elements, faulty_size, cudaMemcpyHostToDevice);
        }
        else{
          printf("plan: Cannot open file, using default setting.\n");
        }
        planFile.close();

        // read faulty bit
        // std::ifstream bitFile("/home/yuhangl/control/bit.txt");
        // Absolute Path
        // fs::path bitPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "bit.txt";
        // Relative Path
        fs::path bitPath = fs::path("./control_" + std::string(job_id) + "/" + std::to_string(gpu_dev)) / "bit.txt";
        std::ifstream bitFile(bitPath);
        if(bitFile.is_open()){
          if (bitFile >> faulty_bit) {
              // std::cout << "faulty_bit = " << faulty_bit << std::endl;
          }
        }
        else{
          printf("bit: Cannot open file, using default setting.\n");
        }
        bitFile.close();

        // current steps
        // Absolute Path
        // fs::path StepPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "current_step.txt";
        // Relative Path
        fs::path StepPath = fs::path("./control_" + std::string(job_id) + "/" + std::to_string(gpu_dev)) / "current_step.txt";
        std::ifstream stepFile(StepPath);
        if (stepFile.is_open()) {
          std::string line;
          if (std::getline(stepFile, line)) {
            // std::cout << line << std::endl;
            // fs::path FIInfoPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "fi_info.txt";
            int step = std::stoi(line);
            std::ofstream ofs(FIInfoPath, std::ios::out | std::ios::app | std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(&step), sizeof(step));

            // std::ofstream ofs(FIInfoPath, std::ios::out | std::ios::app);
            // ofs << std::endl << line << " ";
            // std::cout << std::endl << line << " ";

            ofs.close();
          } 
        }
        stepFile.close();

        // current component
        // Absolute Path
        // fs::path componentPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "component.txt";
        // Relative Path
        fs::path componentPath = fs::path("./control_" + std::string(job_id) + "/" + std::to_string(gpu_dev)) / "component.txt";
        std::ifstream compfile(componentPath);
        if (compfile.is_open()) {
          std::string line;
          if (std::getline(compfile, line)) {
            // std::cout << line << std::endl;
            // fs::path FIInfoPath = fs::path("/home/yuhangl") / ("control_" + std::string(job_id)) / "fi_info.txt";
            std::ofstream ofs(FIInfoPath, std::ios::out | std::ios::app | std::ios::binary);
            ofs.write(line.data(), 2);
            
            // std::ofstream ofs(FIInfoPath, std::ios::out | std::ios::app);
            // ofs << line << " ";
            // std::cout << line << " ";

            ofs.close();
          } 
        }
        compfile.close();
      }
      // std::cout << "faulty_smid = " << faulty_smid << ", faulty_tid = " << faulty_tid << " " << "faulty_bit = " << faulty_bit << std::endl;
    }
    else{
      printf("FI: Cannot open file, using default setting.\n");
    }
    FIFile.close();

    // printf("smem_size: %d\n", smem_size);

    // 0-no split; 1-split; 2-only abft
    // int if_split_phase = 0;
    
    // int *SM_check_res_1;
    cudaMalloc((void**)&SM_check_res_1, 132 * sizeof(int));
    cudaMemset(SM_check_res_1, 0, 132 * sizeof(int));

    bool deBug = false;
    // int iterations = 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t_compute = 0;
    // dim3 new_block(64,1,1);
    dim3 new_grid(12,11,1);

    // void *kernelArgs[] = {&params_, &if_split_phase, &SM_check_res_1, &partion, &faulty_smid, &faulty_tid_1, &faulty_tid_2, &faulty_bit, &d_counter, &d_buf
    //             // &d_all_start, &d_compute, &d_finding, &d_recompute, &d_compare, &d_checking
    //           };

    void *kernelArgs[] = {&params_, &if_split_phase, &SM_check_res_1, &partion, &faulty_smid, &d_faulty_MMAs, &d_faulty_elements, &faulty_bit, &d_counter, &d_buf
                // &d_all_start, &d_compute, &d_finding, &d_recompute, &d_compare, &d_checking
              };

    cutlass::arch::synclog_setup();

    if(DEBUG){
      cudaEventRecord(abft_prepare_end, 0);
      cudaEventSynchronize(abft_prepare_end);
      cudaEventElapsedTime(&t1, abft_prepare_start, abft_prepare_end);
      // printf("myABFT Prepare Time: %f \n", t1);
      destinationFile = "records/time/preparation.txt";
      fullPath = homePath / destinationFile;
      recordTime(fullPath, t1, DEBUG);
    }

    // redirecte stdout
    // int saved_stdout_fd = dup(fileno(stdout));
    // freopen(FIInfoPath.string().c_str(), "a", stdout);

    // cutlass::Kernel<GemmKernel><<<new_grid, block, (smem_size), stream>>>(params_, Signature_Array, 
    //                                                                 Lock_Signature, final_sum, if_split_phase, 
    //                                                                 d_queues, d_SM_JOBS, SM_schedule, SM_check_res,
    //                                                                 d_all_start, d_compute, d_finding, d_recompute, d_compare, d_checking);

    // cudaLaunchCooperativeKernel((void*)cutlass::Kernel_GEMM<GemmKernel>, new_grid, block, kernelArgs, smem_size, stream);
    
    // Grdi: (4, 3, 1); Blocks: (128, 1, 1) when (386, 384, 384)
    // printf("Grdi: (%d, %d, %d); Blocks: (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    cudaDeviceSynchronize();
    if(deBug){
      cudaEventRecord(start, stream);
    }
    // for(int i = 0; i < iterations; i++){
      // cutlass::Kernel<GemmKernel><<<new_grid, block, (smem_size), stream>>>(params_, Signature_Array, 
      //                                                                 Lock_Signature, final_sum, if_split_phase, 
      //                                                                 d_queues, d_SM_JOBS, SM_schedule, SM_check_res,
      //                                                                 d_all_start, d_compute, d_finding, d_recompute, d_compare, d_checking);
      
    cudaLaunchCooperativeKernel((void*)cutlass::Kernel_GEMM<GemmKernel>, new_grid, block, kernelArgs, smem_size, stream);
    // }
    if(deBug){
      cudaEventRecord(stop, stream);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_compute, start, stop);
      printf("compute kernel time: %f\n", t_compute);
    }

    cudaDeviceSynchronize();

    // direct back
    // fflush(stdout);               
    // dup2(saved_stdout_fd, fileno(stdout)); // restore
    // close(saved_stdout_fd);

    if(injection){
      cudaMemcpy(h_buf, d_buf, buf_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_counter, d_counter, 1 * sizeof(int), cudaMemcpyDeviceToHost);

      std::ofstream ofs(FIInfoPath, std::ios::out | std::ios::app | std::ios::binary);
      // std::ofstream ofs(FIInfoPath, std::ios::out | std::ios::app);
      
      ofs.write(reinterpret_cast<const char*>(&h_counter[0]), sizeof(h_counter[0]));
      ofs.write(reinterpret_cast<const char*>(h_buf), sizeof(float) * h_counter[0]);

      // ofs << h_counter[0] << ": ";
      // printf("%d: ", h_counter[0]);
      // int N = (*h_counter) + (*(h_counter+1));
      // for (int i = 0; i < 1*(h_counter[0]); i++) {
      //     ofs << h_buf[i] << " "; 
      //     // if (i != N - 1){
      //     // ofs << " ";  
      //     // }
      //     // printf("%f ", h_buf[i]);
      // }

      // ofs << "|||| ";
      // printf("|||| ");

      // int o = 16*8*2*SM_iter;
      // ofs.write(reinterpret_cast<const char*>(&h_counter[1]), sizeof(h_counter[1]));
      // ofs.write(reinterpret_cast<const char*>(h_buf+o), sizeof(float) * h_counter[1]);

      // ofs << h_counter[1] << ": ";
      // // printf("%d: ", h_counter[1]);
      // for (int i = 0; i < 1*(h_counter[1]); i++) {
      //     ofs << h_buf[i + o] << " "; 
      //     // if (i != 2*(*(h_counter+1)) - 1){
      //     // ofs << " ";  
      //     // }
      //     // printf("%f ", h_buf[o + i]);
      // }

      ofs.close();
    }
    
    result = cudaGetLastError();
    cudaFree(SM_check_res_1);
    
    cudaFree(d_counter);
    cudaFree(d_buf);
    free(h_counter);
    free(h_buf);

    cudaFree(d_faulty_MMAs);
    cudaFree(d_faulty_elements);
    free(h_faulty_MMAs);
    free(h_faulty_elements);

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(int if_split_phase, int partion, bool DEBUG, cudaStream_t stream = nullptr) {
    return run(if_split_phase, partion, DEBUG, stream);
  }
 
  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
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
    /// Instruction-level tile size (concept: GemmShape)
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
    /// If true, kernel supports split-K as a serial reduction
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator_,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout
>
class Gemm<ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
           layout::ColumnMajor,  // partially specialized on LayoutC
           ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
           WarpShape_, InstructionShape_, EpilogueOutputOp_,
           ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial,
           Operator_, GatherA, GatherB, ScatterD, PermuteDLayout> {
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
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static bool const kSplitKSerial = SplitKSerial;

  using UnderlyingOperator = Gemm< 
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
    kAlignmentA,
    SplitKSerial,
    Operator,
    GatherB,
    GatherA,
    ScatterD,
    PermuteDLayout
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using GemmKernel = typename UnderlyingOperator::GemmKernel;
  static int const kAlignmentC = UnderlyingOperator::kAlignmentC;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    // For gather+scatter operations
    int *gather_A_indices;
    int *gather_B_indices;
    int *scatter_D_indices;

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
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ = 
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      int *gather_A_indices_ = nullptr,
      int *gather_B_indices_ = nullptr,
      int *scatter_D_indices_ = nullptr
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      gather_A_indices(gather_A_indices_),
      gather_B_indices(gather_B_indices_),
      scatter_D_indices(scatter_D_indices_) { }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  Gemm() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {
    // printf("Col: stride A (ldA): %d, stride B (ldB): %d, stride C (ldC): %d\n", args.ref_A.stride(0), args.ref_B.stride(0), args.ref_C.stride(0));
    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      {args.ref_A.data(), args.ref_A.stride(0)},
      {args.ref_C.data(), args.ref_C.stride(0)},
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.epilogue,
      args.split_k_slices,
      args.gather_B_indices,
      args.gather_A_indices,
      args.scatter_D_indices
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
  Status run(int if_split_phase, int partion, bool DEBUG, cudaStream_t stream = nullptr) {

    return underlying_operator_.run(if_split_phase, partion, DEBUG, stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(int if_split_phase, int partion, bool DEBUG, cudaStream_t stream = nullptr) {
    return run(if_split_phase, partion, DEBUG, stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    // int if_split_phase, int partion, bool DEBUG, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////