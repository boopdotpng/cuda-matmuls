#include <stdio.h>
#include <vector> 
#include "utils.h"

/*
this kernel accesses B the same way as row. 
*/
__global__ void matmul_B_row_strided(const float *a, const float *b, float *c) {
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= N) return;
  float sum = 0.0f;
  for (uint i = 0; i < N; i++) sum += a[row*N+i] * b[col*N+i];
  c[row*N+col] = sum;
}

__global__ void matmul_B_col_contiguous(const float *a, const float *b ,float *c) {
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  uint col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= N) return;
  float sum = 0.0f;
  for (uint i = 0; i < N; i++) sum += a[row*N+i] * b[i*N+col];
  c[row*N+col] = sum;
}

int main(int argc, const char *argv[]) {
  buffers bufs = allocs();
  cudaEvent_t start, stop;
  float gflops, ms;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N+15) / 16, (N+15)/16);
  printf("launching with %d,%d block dim\n", numBlocks.x, numBlocks.y);
  if (argc > 1) {
    cudaEventRecord(start);
    matmul_B_col_contiguous<<<numBlocks, threadsPerBlock>>>(bufs.A, bufs.B, bufs.C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    gflops = calc_gflops(ms); 
  } 
  else {
    cudaEventRecord(start);
    matmul_B_row_strided<<<numBlocks, threadsPerBlock>>>(bufs.A, bufs.B_t, bufs.C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    gflops = calc_gflops(ms); 
  } 

  bool valid = cpu_val(bufs);
  if (valid) printf("%s: %.2f gflops \n", (argc>1) ? "matmul b col contiguous" : "matmul b row strided", gflops);
  else printf("wrong.\n"); 
  return 0;
}