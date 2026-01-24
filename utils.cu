#include "utils.h"
#include <cmath>
#include <cstdio>
#include <cuda_profiler_api.h>

template<typename T>
std::vector<T> read_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open: " + path);
  size_t size = file.tellg();
  file.seekg(0);
  std::vector<T> buf(size / sizeof(T));
  file.read(reinterpret_cast<char*>(buf.data()), size);
  return buf;
}
template std::vector<float> read_binary<float>(const std::string&);
template std::vector<half> read_binary<half>(const std::string&);

template<typename T>
bool validate(const T *gpu, const float *ref, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float g = static_cast<float>(gpu[i]), r = ref[i];
    float err = fabsf(g - r), thr = TOL * fmaxf(1.0f, fabsf(r));
    if (err > thr) { printf("expected %f got %f at %zu\n", r, g, i); return false; }
  }
  return true;
}
template bool validate<float>(const float*, const float*, size_t);
template bool validate<half>(const half*, const float*, size_t);

template<typename T>
buffers<T> allocs(const std::string& dir) {
  buffers<T> b;
  size_t sz = N * N * sizeof(T);
  cudaMalloc(&b.A, sz); cudaMalloc(&b.B, sz); cudaMalloc(&b.C, sz); cudaMalloc(&b.B_t, sz);
  auto a = read_binary<T>(dir + "/A.bin");
  auto B = read_binary<T>(dir + "/B.bin");
  auto bt = read_binary<T>(dir + "/B_t.bin");
  b.staging_c = read_binary<float>(dir + "/C.bin");
  cudaMemcpy(b.A, a.data(), sz, cudaMemcpyHostToDevice);
  cudaMemcpy(b.B, B.data(), sz, cudaMemcpyHostToDevice);
  cudaMemcpy(b.B_t, bt.data(), sz, cudaMemcpyHostToDevice);
  return b;
}
template buffers<float> allocs<float>(const std::string&);
template buffers<half> allocs<half>(const std::string&);

template<typename KernelFunc, typename T>
float benchmark_kernel(KernelFunc kernel, dim3 grid, dim3 block, T* a, T* b, T* c, std::vector<float>& ref, const char* name) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  // warmup + validate first run
  kernel<<<grid, block>>>(a, b, c);
  cudaDeviceSynchronize();
  std::vector<T> gpuC(N * N);
  cudaMemcpy(gpuC.data(), c, N * N * sizeof(T), cudaMemcpyDeviceToHost);
  if (!validate(gpuC.data(), ref.data(), N * N)) { printf("%s: FAIL\n", name); return 0.0f; }
  for (int i = 0; i < 9; ++i) { kernel<<<grid, block>>>(a, b, c); cudaDeviceSynchronize(); }

  // benchmark 5 runs
  float total = 0.0f, ms;
  for (int i = 0; i < 5; ++i) {
    if (i == 4) cudaProfilerStart();
    cudaEventRecord(start);
    kernel<<<grid, block>>>(a, b, c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (i == 4) cudaProfilerStop();
    total += (2.0 * N * N * N / 1e12) / (ms / 1000.0);
  }
  cudaEventDestroy(start); cudaEventDestroy(stop);
  printf("%s: %.2f TFLOPS\n", name, total / 5.0f);
  return total / 5.0f;
}

template float benchmark_kernel<void(*)(const float*, const float*, float*), float>(
  void(*)(const float*, const float*, float*), dim3, dim3, float*, float*, float*, std::vector<float>&, const char*);
template float benchmark_kernel<void(*)(const half*, const half*, half*), half>(
  void(*)(const half*, const half*, half*), dim3, dim3, half*, half*, half*, std::vector<float>&, const char*);
