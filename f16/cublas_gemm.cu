#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "../utils.h"

int main() {
  auto a = read_binary<half>("./data/f16/A.bin");
  auto b = read_binary<half>("./data/f16/B.bin");
  auto ref = read_binary<float>("./data/f16/C.bin");

  half *dA, *dB; float *dC;
  size_t szh = N * N * sizeof(half), szf = N * N * sizeof(float);
  cudaMalloc(&dA, szh); cudaMalloc(&dB, szh); cudaMalloc(&dC, szf);
  cudaMemcpy(dA, a.data(), szh, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, b.data(), szh, cudaMemcpyHostToDevice);

  cublasHandle_t h; cublasCreate(&h);
  cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
  float alpha = 1.0f, beta = 0.0f;

  auto gemm = [&]() {
    return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
      dB, CUDA_R_16F, N, dA, CUDA_R_16F, N, &beta, dC, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  };

  gemm(); cudaDeviceSynchronize();

  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  const int iters = 100;
  cudaEventRecord(t0);
  for (int i = 0; i < iters; i++) gemm();
  cudaEventRecord(t1); cudaEventSynchronize(t1);

  float ms; cudaEventElapsedTime(&ms, t0, t1);
  float avg = ms / iters;

  std::vector<float> gpuC(N * N);
  cudaMemcpy(gpuC.data(), dC, szf, cudaMemcpyDeviceToHost);
  bool ok = validate(gpuC.data(), ref.data(), N * N);

  printf("cuBLAS HGEMM: %.3f ms, %.2f TFLOPS, valid: %s\n",
    avg, (2.0 * N * N * N / 1e12) / (avg / 1000.0), ok ? "YES" : "NO");

  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cublasDestroy(h); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
