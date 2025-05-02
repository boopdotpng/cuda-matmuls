#include <stdio.h>

__global__ void record_thread_coords(int* coords, int width) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x; // 0 to 3
    int gidy = blockIdx.y * blockDim.y + threadIdx.y; // 0 to 3
    // 16 total threads
    int idx = gidy * width + gidx; // global thread position

    // 2* because we launch 16 threads to cover 32 elements. 
    // the last element would be 2 * 15 + 1 = 31
    // each thread writes two values in the array
    coords[2 * idx + 0] = gidx;
    coords[2 * idx + 1] = gidy;
}

int main() {
  int width = 4, height = 4;
  int* d_coords; // shape: (2,4*4) flattened
  // a 4 by 4 grid where each item is a 2d coordinate
  cudaMalloc(&d_coords, sizeof(int) * 2 * width * height);
  // launch with 4 threads per block (2,2) and 4 total blocks (2,2)
  record_thread_coords<<<dim3(2,2), dim3(2,2)>>>(d_coords, width);

  // output array
  int h_coords[2 * width * height];
  cudaMemcpy(h_coords, d_coords, sizeof(h_coords), cudaMemcpyDeviceToHost);

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      int idx = y * width + x;
      printf("(%d,%d) ", h_coords[2*idx+0], h_coords[2*idx+1]);
    }
    printf("\n");
  }

  return 0;
}