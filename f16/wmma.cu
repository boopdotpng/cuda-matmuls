#include <cuda_fp16.h>
#include <mma.h>
#include "../utils.h"

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
__global__ void wmma_32x32_f16f16_f32acc(
  const half* __restrict__ A,
  const half* __restrict__ B,
  half* __restrict__ C
) {
  // block covers a 32x32 tile of C
  int block_row = blockIdx.y * TILE_M;
  int block_col = blockIdx.x * TILE_N;

  int tid = threadIdx.x;           // 0..127
  int warp_id = tid >> 5;          // 0..3
  // warp tile coords inside the 32x32 block tile
  int warp_row = (warp_id >> 1);   // 0..1
  int warp_col = (warp_id & 1);    // 0..1

  // Shared memory tiles:
  //  A: 32x16 (row-major), B: we store as [Ntile][Ktile] so it can be read as col-major KxN with ld=16
  __shared__ __align__(16) half As[TILE_M][WMMA_K];   // 32x16
  __shared__ __align__(16) half Bs[TILE_N][WMMA_K];   // 32x16  (note: [n][k])
  __shared__ __align__(16) float Cs[TILE_M][TILE_N];  // 32x32

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  for (int k0 = 0; k0 < N; k0 += WMMA_K) {
    // Cooperatively load A(32x16) and B(16x32) into shared memory.
    // Total elements: 512 + 512 = 1024 halfs. 128 threads -> 8 elements/thread.
    for (int i = tid; i < 1024; i += blockDim.x) {
      if (i < 512) {
        int m = i / WMMA_K;        // 0..31
        int k = i % WMMA_K;        // 0..15
        int g_row = block_row + m;
        int g_col = k0 + k;
        As[m][k] = (g_row < N && g_col < N) ? A[g_row * N + g_col] : __float2half(0.0f);
      } else {
        int j = i - 512;
        int k = j / TILE_N;        // 0..15
        int n = j % TILE_N;        // 0..31
        int g_row = k0 + k;
        int g_col = block_col + n;
        // Store as Bs[n][k] so the underlying layout matches col-major (k + n*16)
        // for wmma::col_major with ld=16.
        Bs[n][k] = (g_row < N && g_col < N) ? B[g_row * N + g_col] : __float2half(0.0f);
      }
    }
    __syncthreads();

    // Each warp loads its 16x16 A and 16x16 B subtile and does MMA
    const half* a_ptr = &As[warp_row * WMMA_M][0];     // 16 rows, ld=16
    const half* b_ptr = &Bs[warp_col * WMMA_N][0];     // 16 cols worth, ld=16 (col-major view)

    wmma::load_matrix_sync(a_frag, a_ptr, WMMA_K);
    wmma::load_matrix_sync(b_frag, b_ptr, WMMA_K);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    __syncthreads();
  }

  // Store each warp's 16x16 accumulator tile to shared memory (row-major, ld=32)
  wmma::store_matrix_sync(
    &Cs[warp_row * WMMA_M][warp_col * WMMA_N],
    acc_frag,
    TILE_N,
    wmma::mem_row_major
  );
  __syncthreads();

  // Write out C as half
  for (int i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
    int m = i / TILE_N;
    int n = i % TILE_N;
    int g_row = block_row + m;
    int g_col = block_col + n;
    if (g_row < N && g_col < N) {
      C[g_row * N + g_col] = __float2half(Cs[m][n]);
    }
  }
}

int main(int argc, const char *argv[]) {
  (void)argc;
  (void)argv;
  buffers<half> bufs = allocs<half>("./data/f16");
  dim3 threadsPerBlock(128);
  dim3 numBlocks((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);

  benchmark_kernel(
    wmma_32x32_f16f16_f32acc,
    numBlocks,
    threadsPerBlock,
    bufs.A,
    bufs.B,
    bufs.C,
    bufs.staging_c,
    "wmma 32x32"
  );
  return 0;
}
