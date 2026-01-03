#include <stdio.h>
#include "../utils.h"

#define TILE 32

__global__ void __launch_bounds__(1024) tiled_matmul(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    const uint row = blockIdx.y * TILE + threadIdx.y;
    const uint col = blockIdx.x * TILE + threadIdx.x;
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += TILE) {
      Asub[threadIdx.y][threadIdx.x] = a[row * N + (k0 + threadIdx.x)];
      Bsub[threadIdx.y][threadIdx.x] = b[(k0 + threadIdx.y) * N + col];
      __syncthreads();

      #pragma unroll
      for (int k = 0; k < TILE; ++k)
        acc += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];

      __syncthreads();
    }

    c[row * N + col] = acc;
}

int main(int argc, const char *argv[]) {
    buffers<float> bufs = allocs<float>("./data/f32");
    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks(N / TILE, N / TILE);

    benchmark_kernel(tiled_matmul, numBlocks, threadsPerBlock, bufs.A, bufs.B, bufs.C, bufs.staging_c, "tiled");

    return 0;
}
