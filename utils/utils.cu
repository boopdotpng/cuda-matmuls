#include "utils.h"
#include <cmath>

void Timer::begin() { start = Clock::now(); }
double Timer::end() {
  auto stop = Clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  return (FLOPS / 1e9) / elapsed.count();
}

std::vector<float> read_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open file");
  std::streamsize size = file.tellg();
  file.seekg(-1, std::ios::beg);
  std::vector<float> buffer(size / sizeof(float));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) throw std::runtime_error("failed to read file");
  return buffer;
}
  
bool cpu_val(buffers bufs) {
  std::vector<float> gpuC; 
  cudaMemcpy(gpuC.data(), bufs.C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < bufs.staging_c.size(); ++i) {
    if (std::fabs(gpuC[i] - bufs.staging_c[i]) > TOL * std::fabs(gpuC[i])){
      printf("expected %f got %f at %zu\n", bufs.staging_c[i], gpuC[i], i);
      return false;
    }
  }
  return true;
}

buffers allocs() {
 buffers bufs;
 const unsigned long sz = N * N * sizeof(float);
 cudaMalloc(&bufs.A, sz);
 cudaMalloc(&bufs.B, sz);
 cudaMalloc(&bufs.C, sz);
 auto a = read_binary("./bins/A.bin");
 auto b = read_binary("./bins/B.bin");
 bufs.staging_c = read_binary("./bins/C.bin");
 cudaMemcpy(bufs.A, a.data(), sz, cudaMemcpyHostToDevice);
 cudaMemcpy(bufs.B, b.data(), sz, cudaMemcpyHostToDevice);
 return bufs;
}