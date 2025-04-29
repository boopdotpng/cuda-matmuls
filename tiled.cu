#include <stdio.h>
#include <vector> 
#include <chrono>
#include "utils.h"

__global__ void tiled_matmul(const float *a, const float *b, float *c) {
 
}

int main() {
  buffers bufs = allocs();
  Timer t;

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N+15) / 16, (N+15)/16);
  printf("launching with %d,%d block dim\n", numBlocks.x, numBlocks.y);
  t.begin();
  tiled_matmul<<<numBlocks, threadsPerBlock>>>(bufs.A, bufs.B, bufs.C);
  cudaDeviceSynchronize();
  double gflops = t.end();

  bool valid = cpu_val(bufs);
  if (valid) printf("naive: %.2f gflops \n", gflops);
  else printf("wrong.\n"); 
  return 0;
}