#include <cuda.h> 
#include "../utils.h"

/* 
matmul instruction in ptx:
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {d0, d1, d2, d3};
D -> A * B + C
M=16
N=8
K=16

16*16 A * 16*8 B produces 16*8 output tile in C
A is row major, B is column major

a0..3: 4 32-bit regs holding the packed fp16 fragment of A
b0..1: 2 regs holding the packed fp16 fragment of B 
first d0..3: 4 regs holding fp32 accumulator values for a portion of the 16x8 output tile
last d0..3: technically C: same registers, we're adding whatever is in the accumulator
and storing to the accumulator. C is the accum
*/

/* 
two 4096 x 4096 matrices 
block size (128, 1, 1)	grid size (512, 4, 1) for the cublas gemm
total elements: 16,777,216
2048 blocks of 128 threads each = 262,144 threads launched
every block computes 1024x8 tile 

*/

__global__ __launch_bounds__(128) void matmul_ptx(
  const half* __restrict__ A,
  const half* __restrict__ B,
  half* __restrict__ C
) {
  int lane = threadIdx.x; 
  int warp = threadIdx.y;
  int col0 = blockIdx.x * 8; 
  int row0 = blockIdx.y * 1024;
}

int main(){
  buffers<half> bufs = allocs<half>("./data/f16");
  dim3 threadsPerBlock(128); // 4 warps
  dim3 numBlocks(512, 4, 1);

  benchmark_kernel(
    matmul_ptx, 
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