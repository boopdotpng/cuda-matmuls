#include "utils.h"
#include <cmath>
#include <cstdio>

template<typename T>
std::vector<T> read_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open file: " + path);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<T> buffer(size / sizeof(T));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
    throw std::runtime_error("failed to read file: " + path);
  return buffer;
}

template std::vector<float> read_binary<float>(const std::string& path);
template std::vector<half> read_binary<half>(const std::string& path);

template<typename T>
bool validate(const T *gpu, const T *ref, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float g = static_cast<float>(gpu[i]);
    float r = static_cast<float>(ref[i]);
    float a = fabsf(g - r);
    float thr = TOL * fmaxf(1.0f, fabsf(r));
    if (a > thr) {
      printf("expected %f got %f at %zu\n", r, g, i);
      return false;
    }
  }
  return true;
}

template bool validate<float>(const float *gpu, const float *ref, size_t n);
template bool validate<half>(const half *gpu, const half *ref, size_t n);

template<typename T>
buffers<T> allocs(const std::string& data_dir) {
  buffers<T> bufs;
  const unsigned long sz = N * N * sizeof(T);
  cudaMalloc(&bufs.A, sz);
  cudaMalloc(&bufs.B, sz);
  cudaMalloc(&bufs.C, sz);
  cudaMalloc(&bufs.B_t, sz);

  auto a = read_binary<T>(data_dir + "/A.bin");
  auto b = read_binary<T>(data_dir + "/B.bin");
  auto b_t = read_binary<T>(data_dir + "/B_t.bin");
  bufs.staging_c = read_binary<T>(data_dir + "/C.bin");

  cudaMemcpy(bufs.A, a.data(), sz, cudaMemcpyHostToDevice);
  cudaMemcpy(bufs.B, b.data(), sz, cudaMemcpyHostToDevice);
  cudaMemcpy(bufs.B_t, b_t.data(), sz, cudaMemcpyHostToDevice);
  return bufs;
}

template buffers<float> allocs<float>(const std::string& data_dir);
template buffers<half> allocs<half>(const std::string& data_dir);

template<typename KernelFunc, typename T>
float benchmark_kernel(KernelFunc kernel, dim3 grid, dim3 block, T* a, T* b, T* c, std::vector<T>& ref, const char* name) {
  cudaEvent_t start, stop;
  float ms;
  bool validated = false;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 10 warmup runs
  for (int i = 0; i < 10; ++i) {
    kernel<<<grid, block>>>(a, b, c);
    cudaDeviceSynchronize();

    // Validate only the first run
    if (i == 0 && !validated) {
      std::vector<T> gpuC(N * N);
      cudaMemcpy(gpuC.data(), c, N * N * sizeof(T), cudaMemcpyDeviceToHost);
      if (!validate(gpuC.data(), ref.data(), N * N)) {
        printf("%s: validation failed\n", name);
        return 0.0f;
      }
      validated = true;
    }
  }

  // 5 real runs
  float total_tflops = 0.0f;
  for (int i = 0; i < 5; ++i) {
    cudaEventRecord(start);
    kernel<<<grid, block>>>(a, b, c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    float tflops = (2.0 * N * N * N / 1e12) / (ms / 1000.0);
    total_tflops += tflops;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float avg_tflops = total_tflops / 5.0f;
  printf("%s: %.2f TFLOPS\n", name, avg_tflops);
  return avg_tflops;
}

// Explicit instantiations
template float benchmark_kernel<void(*)(const float*, const float*, float*), float>(
    void(*)(const float*, const float*, float*), dim3, dim3, float*, float*, float*, std::vector<float>&, const char*);
template float benchmark_kernel<void(*)(const half*, const half*, half*), half>(
    void(*)(const half*, const half*, half*), dim3, dim3, half*, half*, half*, std::vector<half>&, const char*);
