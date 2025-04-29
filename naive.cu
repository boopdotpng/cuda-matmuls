#include <stdio.h>
#include <vector> 
#include <chrono>
#include "utils.h"

__global__ void matmul(const float *a, const float *b, float *c) {
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= N) return;
  float sum = 0.0f;
  for (uint i = 0; i < N; i++) sum += a[row*N+i] * b[col+i*N];
  c[row*N+col] = sum;
}

int main() {
  buffers bufs = allocs();
  Timer t;

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N+15) / 16, (N+15)/16);
  printf("launching with %d,%d block dim\n", numBlocks.x, numBlocks.y);
  t.begin();
  matmul<<<numBlocks, threadsPerBlock>>>(bufs.A, bufs.B, bufs.C);
  cudaDeviceSynchronize();
  double gflops = t.end();

  bool valid = cpu_val(bufs);
  if (valid) printf("naive: %.2f gflops \n", gflops);
  else printf("wrong.\n"); 
  return 0;
}