#include <stdio.h>
#include <vector>
#include "utils.h"

__global__ void matmul_B_row_strided(const float *a, const float *b, float *c) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (uint i = 0; i < N; ++i)
        sum += a[row * N + i] * b[col * N + i];
    c[row * N + col] = sum;
}

__global__ void matmul_B_col_contiguous(const float *a, const float *b, float *c) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (uint i = 0; i < N; ++i)
        sum += a[row * N + i] * b[i * N + col];
    c[row * N + col] = sum;
}

int main(int argc, const char *argv[]) {
    buffers bufs = allocs();
    cudaEvent_t start, stop;
    float ms, times[RUNS], sum = 0.0f, avg_gflops;
    const char *name = (argc > 1)
        ? "matmul b col contiguous"
        : "matmul b row strided";
    dim3 tpb(16, 16), nb((N + 15) / 16, (N + 15) / 16);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        if (argc > 1)
            matmul_B_col_contiguous<<<nb, tpb>>>(bufs.A, bufs.B, bufs.C);
        else
            matmul_B_row_strided<<<nb, tpb>>>(bufs.A, bufs.B_t, bufs.C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = calc_gflops(ms);
    }

    for (int i = WARMUP; i < RUNS; ++i)
        sum += times[i];
    avg_gflops = sum / (RUNS - WARMUP);

    if (cpu_val(bufs))
        printf("%s avg (%d runs): %.2f gflops\n", name, RUNS-WARMUP, avg_gflops);
    else
        printf("wrong.\n");

    return 0;
}