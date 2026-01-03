#include <stdio.h>
#include "../utils.h"

#define TILE 16

__global__ void matmul_with_B_transposed(const float *a, const float *b, float *c) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (uint i = 0; i < N; ++i)
        sum += a[row * N + i] * b[col * N + i];
    c[row * N + col] = sum;
}

__global__ void matmul_naive_B_untransposed(const float *a, const float *b, float *c) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (uint i = 0; i < N; ++i)
        sum += a[row * N + i] * b[i * N + col];
    c[row * N + col] = sum;
}

int main(int argc, const char *argv[]) {
    buffers<float> bufs = allocs<float>("./data/f32");
    dim3 tpb(TILE, TILE);
    dim3 nb((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    if (argc > 1) {
        benchmark_kernel(matmul_naive_B_untransposed, nb, tpb, bufs.A, bufs.B, bufs.C, bufs.staging_c, "naive (B untransposed)");
    } else {
        benchmark_kernel(matmul_with_B_transposed, nb, tpb, bufs.A, bufs.B_t, bufs.C, bufs.staging_c, "naive (B transposed)");
    }

    return 0;
}
