#include "../utils.h"

#define TILE 16

__global__ void __launch_bounds__(TILE * TILE)
tiled_matmul(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
  const uint row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
  __shared__ float As[TILE][TILE], Bs[TILE][TILE];

  float acc = 0.0f;
  for (int k0 = 0; k0 < N; k0 += TILE) {
    As[threadIdx.y][threadIdx.x] = a[row * N + k0 + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = b[(k0 + threadIdx.y) * N + col];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  c[row * N + col] = acc;
}

int main() {
  auto b = allocs<float>("./data/f32");
  dim3 tpb(TILE, TILE), nb(N / TILE, N / TILE);
  benchmark_kernel(tiled_matmul, nb, tpb, b.A, b.B, b.C, b.staging_c, "tiled");
}
