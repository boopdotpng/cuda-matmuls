#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "../utils.h"

// CUTLASS GEMM configuration for maximum performance on sm_90+
// Using TensorOp with optimal tile sizes for RTX 5090
using Gemm = cutlass::gemm::device::Gemm<
  half,                           // ElementA
  cutlass::layout::RowMajor,      // LayoutA
  half,                           // ElementB
  cutlass::layout::RowMajor,      // LayoutB
  half,                           // ElementC
  cutlass::layout::RowMajor,      // LayoutC
  float,                          // ElementAccumulator
  cutlass::arch::OpClassTensorOp, // OpClass (Tensor Cores)
  cutlass::arch::Sm90,            // Architecture
  cutlass::gemm::GemmShape<256, 128, 64>,  // ThreadblockShape (MxNxK)
  cutlass::gemm::GemmShape<64, 64, 64>,    // WarpShape
  cutlass::gemm::GemmShape<16, 8, 16>,     // InstructionShape (MMA instruction)
  cutlass::epilogue::thread::LinearCombination<
    half, 128 / cutlass::sizeof_bits<half>::value,
    float, float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,  // Stages for software pipelining
  8,  // AlignmentA
  8   // AlignmentB
>;

int main(int argc, const char *argv[]) {
  (void)argc;
  (void)argv;

  buffers<half> bufs = allocs<half>("./data/f16");

  // CUTLASS GEMM arguments
  int M = N;
  int K = N;
  int lda = N;
  int ldb = N;
  int ldc = N;

  float alpha = 1.0f;
  float beta = 0.0f;

  Gemm gemm_op;

  auto benchmark_cutlass = [&]() {
    cutlass::Status status = gemm_op({
      {M, N, K},
      {bufs.A, lda},
      {bufs.B, ldb},
      {bufs.C, ldc},
      {bufs.C, ldc},
      {alpha, beta}
    });

    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM failed");
    }
    return cudaDeviceSynchronize();
  };

  // Warmup
  benchmark_cutlass();

  // Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int num_iters = 100;
  cudaEventRecord(start);
  for (int i = 0; i < num_iters; i++) {
    benchmark_cutlass();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / num_iters;

  // Validate
  cudaMemcpy(bufs.staging_c.data(), bufs.C, N * N * sizeof(half), cudaMemcpyDeviceToHost);
  std::vector<half> ref = read_binary<half>("./data/f16/c_ref.bin");
  bool valid = validate(bufs.staging_c.data(), ref.data(), N * N);

  double flops = 2.0 * N * N * N;
  double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

  printf("CUTLASS GEMM: %.3f ms, %.2f TFLOPS, valid: %s\n",
         avg_ms, tflops, valid ? "YES" : "NO");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
