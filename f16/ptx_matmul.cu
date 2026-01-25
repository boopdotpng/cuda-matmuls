#include <cuda.h>
#include <cuda_fp16.h>
#include "../utils.h"

__device__ __forceinline__ void mma_m16n8k16(
  uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
  uint32_t b0, uint32_t b1,
  float &d0, float &d1, float &d2, float &d3
) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%0, %1, %2, %3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1)
  );
}

// Load A (16x16, row-major in shared) into 4 regs using two ldmatrix.x2 loads:
// left 16x8 (cols 0..7) -> a0,a1, right 16x8 (cols 8..15) -> a2,a3
__device__ __forceinline__ void load_a_fragment_rowmajor_16x16(
  const half *As, int warp_m, int lane,
  uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3
) {
  const half *base = As + warp_m * 16;

  // For .x2, threads 0..15 provide the 16 row addresses. Others can alias safely.
  int r = lane & 15;

  uint32_t addrL = static_cast<uint32_t>(__cvta_generic_to_shared(base + r * 16 + 0));
  uint32_t addrR = static_cast<uint32_t>(__cvta_generic_to_shared(base + r * 16 + 8));

  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
    "{%0, %1}, "
    "[%2];\n"
    : "=r"(a0), "=r"(a1)
    : "r"(addrL)
  );

  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
    "{%0, %1}, "
    "[%2];\n"
    : "=r"(a2), "=r"(a3)
    : "r"(addrR)
  );
}

// Load B (16x8, row-major in shared) using ldmatrix.x2.trans so that the fragment
// is effectively in the col-major form required by mma ... row.col
__device__ __forceinline__ void load_b_fragment_rowmajor_16x8_to_colmajor(
  const half *Bs, int lane,
  uint32_t &b0, uint32_t &b1
) {
  int r = lane & 15;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(Bs + r * 8));

  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
    "{%0, %1}, "
    "[%2];\n"
    : "=r"(b0), "=r"(b1)
    : "r"(addr)
  );
}

// Correct m16n8 accumulator lane->(row,col) mapping for {d0..d3}
__device__ __forceinline__ void store_d_fragment_m16n8(
  half *C, int c_row0, int c_col0, int lane,
  float d0, float d1, float d2, float d3
) {
  int row = lane >> 2;        // 0..7
  int col = (lane & 3) << 1;  // 0,2,4,6

  int r0 = c_row0 + row;
  int r1 = c_row0 + row + 8;
  int c0 = c_col0 + col;

  C[r0 * 4096 + (c0 + 0)] = __float2half(d0);
  C[r0 * 4096 + (c0 + 1)] = __float2half(d1);
  C[r1 * 4096 + (c0 + 0)] = __float2half(d2);
  C[r1 * 4096 + (c0 + 1)] = __float2half(d3);
}

__global__ __launch_bounds__(128) void matmul_ptx_fixed(
  const half *__restrict__ A,
  const half *__restrict__ B,
  half *__restrict__ C
) {
  int lane = threadIdx.x;     // 0..31
  int warp = threadIdx.y;     // 0..3
  int tid  = warp * 32 + lane; // 0..127

  int block_n = (int)blockIdx.x * 8;   // N tile start
  int block_m = (int)blockIdx.y * 64;  // M tile start

  __shared__ __align__(16) half As[64 * 16];  // 64x16 (row-major)
  __shared__ __align__(16) half Bs[16 * 8];   // 16x8  (row-major)

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  for (int kt = 0; kt < 4096; kt += 16) {
    // Load As: 64x16 = 1024 halfs, 128 threads -> 8 halfs per thread
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int idx = tid + i * 128;  // 0..1023
      int r = idx >> 4;         // /16 -> 0..63
      int c = idx & 15;         // %16 -> 0..15
      As[r * 16 + c] = A[(block_m + r) * 4096 + (kt + c)];
    }

    // Load Bs row-major: 16x8 = 128 halfs, 128 threads -> 1 half per thread
    int br = tid >> 3;     // /8 -> 0..15
    int bc = tid & 7;      // %8 -> 0..7
    Bs[br * 8 + bc] = B[(kt + br) * 4096 + (block_n + bc)];

    __syncthreads();

    int warp_m = warp * 16;

    uint32_t a0, a1, a2, a3;
    uint32_t b0, b1;

    load_a_fragment_rowmajor_16x16(As, warp_m, lane, a0, a1, a2, a3);
    load_b_fragment_rowmajor_16x8_to_colmajor(Bs, lane, b0, b1);

    mma_m16n8k16(a0, a1, a2, a3, b0, b1, acc0, acc1, acc2, acc3);

    __syncthreads();
  }

  int c_row0 = block_m + warp * 16;
  int c_col0 = block_n;
  store_d_fragment_m16n8(C, c_row0, c_col0, lane, acc0, acc1, acc2, acc3);
}


int main(){
  buffers<half> bufs = allocs<half>("./data/f16");
  dim3 threadsPerBlock(32, 4); // 4 warps
  dim3 numBlocks(512, 64);

  benchmark_kernel(
    matmul_ptx_fixed, 
    numBlocks, 
    threadsPerBlock,
    bufs.A,
    bufs.B,
    bufs.C,
    bufs.staging_c,
    "ptx wmma"
  );
  return 0;
}
