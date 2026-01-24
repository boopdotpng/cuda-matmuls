#include "../utils.h"

#define TILE 16

__global__ void matmul_Bt(const float *a, const float *b, float *c) {
  uint row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;
  for (uint i = 0; i < N; ++i) sum += a[row * N + i] * b[col * N + i];
  c[row * N + col] = sum;
}

__global__ void matmul_B(const float *a, const float *b, float *c) {
  uint row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;
  for (uint i = 0; i < N; ++i) sum += a[row * N + i] * b[i * N + col];
  c[row * N + col] = sum;
}

int main(int argc, char**) {
  auto b = allocs<float>("./data/f32");
  dim3 tpb(TILE, TILE), nb(N / TILE, N / TILE);
  if (argc > 1) benchmark_kernel(matmul_B, nb, tpb, b.A, b.B, b.C, b.staging_c, "naive");
  else benchmark_kernel(matmul_Bt, nb, tpb, b.A, b.B_t, b.C, b.staging_c, "naive (B transposed)");
}
