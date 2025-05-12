#pragma once
#include <vector>
#include <string>
#include <fstream> 

#define N 4096
#define TILE 32
#define TOL 1e-3
#define RUNS 8
#define WARMUP 2

struct buffers {
  float *A, *B, *B_t, *C; 
  std::vector<float> staging_c; 
};

bool cpu_val(buffers bufs);

buffers allocs();

float calc_gflops(double time);
