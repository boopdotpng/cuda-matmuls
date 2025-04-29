#pragma once
#include <vector>
#include <string>
#include <fstream> 
#include <chrono>

#define N 4096
#define TILE 16
#define TOL 1e-3

struct buffers {
  float *A, *B, *C; 
  std::vector<float> staging_c; // cpu validation buffer
};

// validate cpu results vs exported C value
bool cpu_val(buffers bufs);

// get A and B as cuda pointers
buffers allocs();

// flops calculation and timing 
struct Timer {
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start;
  void begin();
  double end();
};