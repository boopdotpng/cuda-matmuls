#include <stdio.h>
#include <vector>
#include "utils.h"

__global__ void tiled_matmul(const float *a, const float *b, float *c) {
    uint row = blockIdx.y * TILE + threadIdx.y;
    uint col = blockIdx.x * TILE + threadIdx.x;
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += TILE) {
      Asub[threadIdx.y][threadIdx.x] = a[row * N + (k0 + threadIdx.x)];
      Bsub[threadIdx.y][threadIdx.x] = b[(k0 + threadIdx.y) * N + col];
      __syncthreads();

      for (int k = 0; k < TILE; ++k)
        acc += Asub[threadIdx.y][k]   * Bsub[k][threadIdx.x];
      
      __syncthreads();
    }

    c[row * N + col] = acc;
}

int main(int argc, const char *argv[]) {
    buffers bufs = allocs();
    cudaEvent_t start, stop;
    float ms, times[RUNS], sum = 0.0f, avg_gflops;
    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks(N / TILE, N / TILE);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        tiled_matmul<<<numBlocks, threadsPerBlock>>>(bufs.A, bufs.B, bufs.C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = calc_gflops(ms);
    }

    for (int i = WARMUP; i < RUNS; ++i) {
        sum += times[i];
    }
    avg_gflops = sum / (RUNS - WARMUP);

    if (cpu_val(bufs))
        printf("tiled matmul avg (%d runs): %.2f gflops\n", RUNS-WARMUP, avg_gflops);
    else
        printf("wrong.\n");

    return 0;
}