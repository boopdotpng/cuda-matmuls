#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#ifdef __CUDACC__
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
#endif

#define N 4096
#define TOL 1e-3

template<typename T> struct buffers { T *A, *B, *B_t, *C; std::vector<float> staging_c; };
template<typename T> buffers<T> allocs(const std::string& dir);
template<typename T> bool validate(const T *gpu, const float *ref, size_t n);
template<typename T> std::vector<T> read_binary(const std::string& path);
template<typename KernelFunc, typename T>
float benchmark_kernel(KernelFunc kernel, dim3 grid, dim3 block, T* a, T* b, T* c, std::vector<float>& ref, const char* name);
