#include <stdio.h>

__global__ void record_thread_coords(int* coords, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to width (x)
  int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to height (y)

  // 16 total threads
  int idx = row * width + col; // global thread position

  // each thread writes two values in the array
  // the last two threads would write indices 2*23+0=46 and 2*23+1=47
  coords[2 * idx + 0] = col; // x
  coords[2 * idx + 1] = row; // y
}

int main() {
  int width = 4, height = 6;

  int* d_coords;
  cudaMalloc(&d_coords, sizeof(int) * 2 * width * height);

  dim3 block_dim(2, 2); // 2x2 threads per block
  dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                (height + block_dim.y - 1) / block_dim.y);

  record_thread_coords<<<grid_dim, block_dim>>>(d_coords, width);

  int h_coords[2 * width * height];
  cudaMemcpy(h_coords, d_coords, sizeof(h_coords), cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      printf("(%d,%d) ", h_coords[2 * idx + 0], h_coords[2 * idx + 1]);
    }
    printf("\n");
  }

  cudaFree(d_coords);
  return 0;
}