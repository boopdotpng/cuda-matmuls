// benchmark_sgemm.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                    \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

#define CUBLAS_CHECK(call)                                                 \
  do {                                                                     \
    cublasStatus_t status = call;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                 \
      fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,      \
              status);                                                     \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

int main() {
    const int N = 4096;
    const int64_t size = int64_t(N) * N;
    const size_t bytes = size * sizeof(float);

    // allocate host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int64_t i = 0; i < size; ++i) {
        h_A[i] = drand48();
        h_B[i] = drand48();
    }

    // allocate device
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // ensure pure FP32, disable TF32 tensor cores
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    const float alpha = 1.0f, beta = 0.0f;

    // warmup
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // timing
    const int nIter = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float totalMs = 0;
    for (int i = 0; i < nIter; ++i) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_B, N,
            d_A, N,
            &beta,
            d_C, N
        ));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;
    }

    float avgMs = totalMs / nIter;
    double gflops = 2.0 * double(N) * N * N / (avgMs / 1e3) / 1e9;

    printf("Average time over %d runs: %f ms\n", nIter, avgMs);
    printf("Effective performance: %f GFLOPS\n", gflops);

    // cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}


//  nvcc -Wno-deprecated-gpu-targets -O3 benchmark_sgemm.cu -lcublas -o benchmark_sgemm