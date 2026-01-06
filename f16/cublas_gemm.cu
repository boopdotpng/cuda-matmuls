#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "../utils.h"

int main(int argc, const char *argv[]) {
  (void)argc;
  (void)argv;

  const std::string data_dir = "./data/f16";
  auto host_a = read_binary<half>(data_dir + "/A.bin");
  auto host_b = read_binary<half>(data_dir + "/B.bin");
  auto host_c = read_binary<float>(data_dir + "/C.bin");

  half *dA = nullptr;
  half *dB = nullptr;
  float *dC = nullptr;
  const size_t sz_half = N * N * sizeof(half);
  const size_t sz_float = N * N * sizeof(float);
  cudaMalloc(&dA, sz_half);
  cudaMalloc(&dB, sz_half);
  cudaMalloc(&dC, sz_float);
  cudaMemcpy(dA, host_a.data(), sz_half, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, host_b.data(), sz_half, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Enable tensor cores
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto benchmark_cublas = [&]() {
    // Use cublasGemmEx to explicitly control compute type
    // CUBLAS_COMPUTE_32F = f16 inputs with f32 accumulation
    // Row-major C = A * B becomes column-major C^T = B^T * A^T
    // Pass B, A with CUBLAS_OP_N since row-major is already transposed in column-major view
    return cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, N, N,
      &alpha,
      dB, CUDA_R_16F, N,  // B (f16)
      dA, CUDA_R_16F, N,  // A (f16)
      &beta,
      dC, CUDA_R_32F, N,  // C (f32)
      CUBLAS_COMPUTE_32F,     // f32 accumulation!
      CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
  };

  // Warmup
  benchmark_cublas();
  cudaDeviceSynchronize();

  // Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int num_iters = 100;
  cudaEventRecord(start);
  for (int i = 0; i < num_iters; i++) {
    benchmark_cublas();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / num_iters;

  // Validate
  std::vector<float> gpu_c(N * N);
  cudaMemcpy(gpu_c.data(), dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  bool valid = validate(gpu_c.data(), host_c.data(), N * N);

  double flops = 2.0 * N * N * N;
  double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

  printf("cuBLAS HGEMM: %.3f ms, %.2f TFLOPS, valid: %s\n",
         avg_ms, tflops, valid ? "YES" : "NO");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
