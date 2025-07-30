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

#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/util/command_line.h"

#include "helper.h"


#pragma warning( disable : 4503)

/*
This example demonstrates how to use cutlass to compute a batched strided gemm in two different ways:
  1. By specifying pointers to the first matrices of the batch and the stride between the consecutive
    matrices of the batch (this is called a strided batched gemm).
  2. By copying pointers to all matrices of the batch to the device memory (this is called an array gemm).
In this example, both A and B matrix are non-transpose and column major matrix
batched_C = batched_A x batched_B
As an example, matrix C can be seen as
-----------------------------------------------------------
(0,0,0) | (0,0,1) | (0,0,2) | (1,0,0) | (1,0,1) | (1,0,2) |
-----------------------------------------------------------
(0,1,0) | (0,1,1) | (0,1,2) | (1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------------------------------------
(0,2,0) | (0,2,1) | (0,2,2) | (1,2,0) | (1,2,1) | (1,2,2) |
-----------------------------------------------------------
(0,3,0) | (0,3,1) | (0,3,2) | (1,3,0) | (1,3,1) | (1,3,2) |
-----------------------------------------------------------
(0,4,0) | (0,4,1) | (0,4,2) | (1,4,0) | (1,4,1) | (1,4,2) |
-----------------------------------------------------------
(0,5,0) | (0,5,1) | (0,5,2) | (1,5,0) | (1,5,1) | (1,5,2) |
-----------------------------------------------------------
          batch 0          |           batch 1
where we denote each element with (batch_idx, row_idx, column_idx)
In this example, batch size is 2, M is 6 and N is 3
The stride (batch_stride_C) between the first element of two batches is ldc * n

matrix A can be seen as
---------------------------------------
(0,0,0) | (0,0,1) | (1,0,0) | (1,0,1) |
---------------------------------------
(0,1,0) | (0,1,1) | (1,1,0) | (1,1,1) |
---------------------------------------
(0,2,0) | (0,2,1) | (1,2,0) | (1,2,1) |
---------------------------------------
(0,3,0) | (0,3,1) | (1,3,0) | (1,3,1) |
---------------------------------------
(0,4,0) | (0,4,1) | (1,4,0) | (1,4,1) |
---------------------------------------
(0,5,0) | (0,5,1) | (1,5,0) | (1,5,1) |
---------------------------------------
    batch 0      |      batch 1
, where batch size is 2, M is 6 and K is 2
The stride (batch_stride_A) between the first element of two batches is lda * k

matrix B can be seen as
-----------------------------
(0,0,0) | (0,0,1) | (0,0,2) |
----------------------------- batch 0
(0,1,0) | (0,1,1) | (0,1,2) |
-------------------------------------
(1,0,0) | (1,0,1) | (1,0,2) |
----------------------------- batch 1
(1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------
, where the batch size is 2, N is 3 and K is 2
The stride (batch_stride_B) between the first element of two batches is k

nvcc ampere_tf32_batched_gemm.cu -O0 -I/home/yuhangl/cutlass/include -I/home/yuhangl/cutlass/tools/util/include -I/home/yuhangl/cutlass/examples/common -arch=sm_90 -o bout.exe

nvcc ampere_tf32_batched_gemm.cu -O0 -I/home/yuhangl/origin_cutlass/cutlass/include -I/home/yuhangl/origin_cutlass/cutlass/tools/util/include -I/home/yuhangl/origin_cutlass/cutlass/examples/common -arch=sm_90 -o blout.exe

ncu -f -o batch32 --set full ./bout.exe --batch=256 --m=1024 --n=1024 --k=128 --split=0 --iterations=1
*/

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;

  int partition;
  int if_split_phase;

  int validation;
  
  Options():
    help(false),
    problem_size({128*3, 128*3, 128*3}),
    batch_count(1),
    reference_check(true),
    iterations(1),
    alpha(1),
    beta(0),
    partition(),
    if_split_phase(2),
    validation(0) { }

  // bool valid() {
  //   return true;
  // }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("batch", batch_count);

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("split", if_split_phase);

    cmd.get_cmd_line_argument("validate", validation);

    // cmd.get_cmd_line_argument("partition", partition);
    partition = 1;

    // add checksum size
    if(if_split_phase == 1 || if_split_phase == 0){
      problem_size.n() += partition * 2;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "14_ampere_tf32_tensorop_gemm example\n\n"
      << "  This example uses the CUTLASS Library to execute TF32 tensorop GEMM computations.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --batch=<int>               Batched Number\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --partition=<int>           Number of partition of the matrix.\n\n"
      << "  --split=<int>               0-no split, 1-split, 2-baseline.\n\n"
      << "  --validate=<int>            0-no validate, 1-validate";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                      // memory access. For a byte, it's 16
                                                      // elements. This becomes the vector width of
                                                      // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 4; 

cudaError_t cutlass_strided_batched_sgemm(
  int m, 
  int n,
  int k,
  float alpha,
  float const *A,
  int lda,
  long long int batch_stride_A,
  float const *B,
  int ldb,
  long long int batch_stride_B,
  float *C,
  int ldc,
  long long int batch_stride_C,
  float beta,
  int batch_count,
  int if_split_phase,
  int partition) {

  using Gemm = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ShapeMMAOp,
    EpilogueOp
    // SwizzleThreadBlock,
    // NumStages
  >;

  // typename Gemm::Arguments arguments{
  //     {m, n, k},
  //     {A, lda}, 
  //     batch_stride_A,
  //     {B, ldb}, 
  //     batch_stride_B,
  //     {C, ldc}, 
  //     batch_stride_C,
  //     {C, ldc}, 
  //     batch_stride_C,
  //     {alpha, beta},
  //     batch_count
  // };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  // size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;

  // Check the problem size is supported or not 
  // cutlass::Status status = gemm_op.can_implement(arguments);
  // CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  // status = gemm_op.initialize(arguments, workspace.get());
  // CUTLASS_CHECK(status);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count},
    if_split_phase, partition,
    stream
  );

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

template<typename T> 
cudaError_t strided_batched_gemm_nn_reference(
  int m,
  int n,
  int k,
  T alpha,
  std::vector<T> const &A, 
  int lda,
  long long int batch_stride_A,
  std::vector<T> const &B, 
  int ldb,
  long long int batch_stride_B,
  std::vector<T> &C, 
  int ldc,
  long long int batch_stride_C,
  T beta,
  int batch_count) {
  /*
  strided batched gemm NN
  */
  
  cudaError_t result = cudaSuccess;

  if (A.size() < size_t(lda * k * batch_count)) {
    std::cout << "the size of A is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  if (B.size() < size_t(ldb * n)) {
    std::cout << "the size of B is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  if (C.size() < size_t(ldc * n * batch_count)) {
    std::cout << "the size of C is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  
  for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
        T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
        for (int k_idx = 0; k_idx < k; k_idx++) {
          accum += alpha 
            * A[batch_idx * batch_stride_A + k_idx * lda + m_idx]
            * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
        }
        C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
      }
    }
  }

  return result;
}

template<typename T>
bool valid( int m, int n, int k,
            std::vector<T> &C, std::vector<T> &ref_C, 
            int ldc, long long int batch_stride_C, int batch_count){
  bool correct = true;
  for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
          T c = C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
          T ref_c = ref_C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
          if(c != ref_c){
            printf("batch: %d, m: %d, n: %d, C: %f, ref_C: %f, diff: %f\n", batch_idx, m_idx, n_idx, c, ref_c, (c-ref_c));
            correct = false;
          }
      }
    }
  }
  return correct;
}

template<typename T>
void outputChk(std::vector<T> &A, int64_t nb, int64_t ld, int64_t stride, int64_t row, int64_t col){
  for(int i = 0; i < nb; i++){
    printf("[ \n");
    for(int r = 0; r < row; r++){
      printf("|");
      for(int c = 0; c < col; c++){
        printf("%.6f", (float)(A[i*stride + c*ld + r]));
        printf(", ");
      }
      printf("!\n");
    }
    printf("]\n");
  }
}

template <typename Element>
void encode_col_checksum(std::vector<Element> &A, int k, int n, int partition, int b_idx, int stride, int lda){
  int m = 1;
  // init checksum vector
  float *chk_vector;
  chk_vector = (float*)malloc(sizeof(float)* k * 1);
  for(int c = 0; c < k; c++){
    chk_vector[c] = (float)1;
    // chk_vector[c + k] = (float)(c+1);
  }
  // encode chksum - column major
  int k_per_partion = k / partition;
  for(int p = 0; p < partition; p++){
    for(int c = 0; c < n; c++){
      for(int r = 0; r < m; r++){
        float sum = 0.0f;
        for(int i = 0; i < k_per_partion; i++){
          float a = chk_vector[r + i * m];
          float b = A[(i + k_per_partion * p) + c * lda + (b_idx * stride)];
          sum += a * b;
        }
        int idx = (k + p) + (c * lda) + (b_idx * stride);
        A[idx] = sum;
      }
    }
  }
  free(chk_vector);
}

template <typename Element>
void encode_row_checksum(std::vector<Element> &A, int m, int k, int partition, int b_idx, int stride, int lda){
  int n = 2;
  int k_per_partion = k / partition;

  // init checksum vector
  float *chk_vector;
  chk_vector = (float*)malloc(sizeof(Element)* k_per_partion * n);
  for(int r = 0; r < k_per_partion; r++){
    chk_vector[r] = (Element)1;
    chk_vector[k_per_partion + r] = (Element)(r + 1);
  }

  // encode row chksum - column major
  for(int p = 0; p < partition; p++){
    for(int c = 0; c < n; c++){
      for(int r = 0; r < m; r++){
        float sum = 0.0f;
        for(int i = 0; i < k_per_partion; i++){
          float a = A[r + (i + k_per_partion * p) * lda + (b_idx * stride)];
          float b = chk_vector[i + c * k_per_partion];
          sum += a * b;
        }
        // int idx = (k + p) + (c * lda) + (b_idx * stride);
        int idx = (lda * (k + 2*p)) + (r + c * m + (b_idx * stride));
        A[idx] = sum;
      }
    }
  }
  free(chk_vector);
}

cudaError_t run_batched_gemm(bool use_array, Options &options) {

  const char* gemm_desc = use_array ? "array" : "strided batched";
  std::cout << "Running " << gemm_desc << " gemm" << std::endl;

  // Arbitrary problem size
  int const m = options.problem_size.m();
  // int const n = options.problem_size.n();
  int const k = options.problem_size.k();
  int const batch_count = options.batch_count;

  // A, B are non-transpose, column major
  int const lda = m;
  // int const ldb = k * batch_count;
  int const ldb = k;
  int const ldc = m;

  int const count_A = batch_count * lda * k;
  // int const count_B = ldb * n;
  int const count_B = batch_count * ldb * options.problem_size.n();
  int const count_C = batch_count * ldc * options.problem_size.n();

  // the memory is batched along K dimension
  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  // long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(ldb) * static_cast<long long int>(options.problem_size.n());
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(options.problem_size.n());

  // alpha and beta
  float alpha = 1.f;
  float beta = 0.f;

  cudaError_t result = cudaSuccess;

  // allocate the host memory
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);
  std::vector<float> result_C(count_C);

  // allocate the device memory
  float *A;
  float *B;
  float *C;

  result = cudaMalloc(&A, count_A * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&B, count_B * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&C, count_C * sizeof(float));
  if (result != cudaSuccess) {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }

  // Limit range to avoid floating-point errors
  int const kRange = 8;
  float const DIV = 1;

  // fill A
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < k; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_A[row_idx + col_idx * lda + b_idx * lda * k] = static_cast<float>((row_idx + col_idx * lda + b_idx * lda * k) % kRange) / DIV;
        // host_A[row_idx + col_idx * lda + b_idx * lda * k] = 1.f;
      }
    }
    // if(options.if_split_phase == 1 || options.if_split_phase == 0){
    //   encode_col_checksum(host_A, (m-1*options.partition), k, options.partition, b_idx, batch_stride_A, lda);
    // }
  }
  // outputChk(host_A, batch_count, lda, batch_stride_A, m, k);
  
  // fill B
  int n1 = options.problem_size.n();
  if(options.if_split_phase == 1 || options.if_split_phase == 0){
    n1 = options.problem_size.n() - 2 * options.partition;
  }

  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n1; col_idx++) {
      for (int row_idx = 0; row_idx < k; row_idx++) {
        // n = n, k = k, ldb = k * batch_count, 
        // host_B[row_idx + col_idx * ldb + b_idx * k] = static_cast<float>(((n + k * ldb + batch_count * k) - (row_idx + col_idx * ldb + b_idx * k)) % kRange);
        // host_B[row_idx + col_idx * ldb + b_idx * ldb * options.problem_size.n()] = static_cast<float>(((options.problem_size.n() + k * ldb + batch_count * k) - (row_idx + col_idx * ldb + b_idx * k)) % kRange) / DIV;
        
        host_B[row_idx + col_idx * ldb + b_idx * ldb * options.problem_size.n()] = static_cast<float>((row_idx + col_idx * ldb + b_idx * ldb * options.problem_size.n()) % kRange) / DIV;
        // host_B[row_idx + col_idx * ldb + b_idx * ldb * options.problem_size.n()] = 1.f;
      }
    }
    if(options.if_split_phase == 1 || options.if_split_phase == 0){
      encode_row_checksum(host_B, k, (options.problem_size.n() - 2 * options.partition), options.partition, b_idx, batch_stride_B, ldb);
    }
  }
  // outputChk(host_B, batch_count, ldb, batch_stride_B, k, options.problem_size.n());

  // fill C
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < options.problem_size.n(); col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_C[row_idx + col_idx * ldc + b_idx * ldc * options.problem_size.n()] = 0.f;
      }
    }
  }

  printf("----finish filling matrices-------\n");

  // ref memory
  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);
  // copy host memory to device
  result = cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  int const n = (options.if_split_phase == 1 || options.if_split_phase == 0) ? (options.problem_size.n() - 2 * options.partition) : options.problem_size.n();

  // run cutlass
  for(int i = 0; i < options.iterations; i++){
    result = cutlass_strided_batched_sgemm(
      m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C,
      beta, batch_count, options.if_split_phase, options.partition);
    if (result != cudaSuccess){
      std::cerr << "cutlass result = " << result << std::endl;
      return result;
    }
  }
  
  // copy device memory to host
  result = cudaMemcpy(result_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  // printf("C:\n");
  // outputChk(result_C, batch_count, ldc, batch_stride_C, m, options.problem_size.n());

  //compare with reference code
  if(options.validation == 1){
    result = strided_batched_gemm_nn_reference(m, options.problem_size.n(), k, alpha, ref_A, lda, batch_stride_A, ref_B, ldb, batch_stride_B, ref_C, ldc, batch_stride_C,
      beta, batch_count);

    // bool res = valid(m, options.problem_size.n(), k, result_C, ref_C, ldc, batch_stride_C, batch_count);
    // if(res){
    //   printf("self-validate not error\n");
    // }
    // else{
    //   printf("self-validate error detected\n");
    // }

    // if (result != 0){
    //   std::cerr << "reference result = " << result << std::endl;
    //   return result;
    // }
  }

  // printf("ref C:\n");
  // outputChk(ref_C, batch_count, ldc, batch_stride_C, m, options.problem_size.n());

  // Expect bit-level accuracy for this simple example
  if (ref_C != result_C) {
    std::cout << "CUTLASS " << gemm_desc << " gemm does not run correctly" << std::endl;
    return cudaErrorUnknown;
  }

  // free memory
  result = cudaFree(A);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(B);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(C);
  if (result != cudaSuccess) {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }

  return result;
}

int main(int argc, const char **argv) {

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d x %d TF32 tensor op Matrix Multiply\n", \
    options.batch_count, options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  cudaError_t result = cudaSuccess;
  for (bool use_array : {false}) {
    result = run_batched_gemm(use_array, options);
    if (result == cudaSuccess) {
      std::cout << "Passed." << std::endl;
    } else {
      break;
    }
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}
