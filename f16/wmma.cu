#include <cuda_fp16.h>
#include <mma.h>
#include "../utils.h"
using namespace nvcuda;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int TILE_M = 32, TILE_N = 32;

__global__ void wmma_32x32(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C) {
  int block_row = blockIdx.y * TILE_M, block_col = blockIdx.x * TILE_N;
  int tid = threadIdx.x, warp_id = tid >> 5;
  int warp_row = warp_id >> 1, warp_col = warp_id & 1;

  __shared__ __align__(16) half As[TILE_M][WMMA_K];   // 32x16
  __shared__ __align__(16) half Bs[TILE_N][WMMA_K];   // 32x16 stored as [n][k]
  __shared__ __align__(16) float Cs[TILE_M][TILE_N];  // 32x32

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k0 = 0; k0 < N; k0 += WMMA_K) {
    for (int i = tid; i < 1024; i += blockDim.x) {
      if (i < 512) {
        int m = i / WMMA_K, k = i % WMMA_K;
        int gr = block_row + m, gc = k0 + k;
        As[m][k] = (gr < N && gc < N) ? A[gr * N + gc] : __float2half(0.0f);
      } else {
        int j = i - 512, k = j / TILE_N, n = j % TILE_N;
        int gr = k0 + k, gc = block_col + n;
        Bs[n][k] = (gr < N && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
      }
    }
    __syncthreads();
    wmma::load_matrix_sync(a_frag, &As[warp_row * WMMA_M][0], WMMA_K);
    wmma::load_matrix_sync(b_frag, &Bs[warp_col * WMMA_N][0], WMMA_K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }

  wmma::store_matrix_sync(&Cs[warp_row * WMMA_M][warp_col * WMMA_N], acc, TILE_N, wmma::mem_row_major);
  __syncthreads();

  for (int i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
    int m = i / TILE_N, n = i % TILE_N;
    int gr = block_row + m, gc = block_col + n;
    if (gr < N && gc < N) C[gr * N + gc] = __float2half(Cs[m][n]);
  }
}

int main() {
  auto b = allocs<half>("./data/f16");
  dim3 tpb(128), nb((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);
  benchmark_kernel(wmma_32x32, nb, tpb, b.A, b.B, b.C, b.staging_c, "wmma 32x32");
}
