#include <stdio.h>
#include <vector> 
#include "utils.h"

__global__ void tiled_matmul(const float *a, const float *b ,float *c) {
  // 2d dispatch: 256,256 blocks, 16x16 threads per block
  uint row = blockIdx.y * TILE + threadIdx.y;
  uint col = blockIdx.x * TILE + threadIdx.x;
  __shared__ float Asub[TILE][TILE];
  __shared__ float Bsub[TILE][TILE];

  float acc = 0.0f;
  for (int k0 = 0; k0 < N; k0 += TILE) {
    // load one tile each from A and B into shared memory 
    Asub[threadIdx.y][threadIdx.x] = a[row * N + (k0 + threadIdx.x)];
    Bsub[threadIdx.y][threadIdx.x] = b[(k0 + threadIdx.y) * N + col];

    // sync threads in block 
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];

    __syncthreads();
  }

  c[row*N+col] = acc;
}

int main(int argc, const char *argv[]) {
  buffers bufs = allocs();
  cudaEvent_t start, stop;
  float gflops, ms;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 threadsPerBlock(TILE, TILE);
  // how many 16x16 blocks in a 4096x4096 matrix? 
  dim3 numBlocks(N/TILE, N/TILE);
  printf("launching with %d,%d block dim\n", numBlocks.x, numBlocks.y);
  cudaEventRecord(start);
  tiled_matmul<<<numBlocks, threadsPerBlock>>>(bufs.A, bufs.B, bufs.C);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop); 
  gflops = calc_gflops(ms); 

  bool valid = cpu_val(bufs);
  if (valid) printf("tiled matmul: %.2f gflops \n", gflops);
  else printf("wrong.\n"); 
  return 0;
}