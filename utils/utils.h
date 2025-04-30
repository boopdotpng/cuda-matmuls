#pragma once
#include <vector>
#include <string>
#include <fstream> 

#define N 4096
#define TILE 16
#define TOL 1e-3
#define RUNS 8
#define WARMUP 2

struct buffers {
  float *A, *B, *B_t, *C; 
  std::vector<float> staging_c; // cpu validation buffer
};

// validate cpu results vs exported C value
bool cpu_val(buffers bufs);

// get A and B as cuda pointers
buffers allocs();

float calc_gflops(double time);
