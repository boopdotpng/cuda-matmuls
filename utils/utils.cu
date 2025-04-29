#include "utils.h"
#include <cmath>
#include <immintrin.h>

float calc_gflops(double time) {
  return (2.0 * N * N * N / 1e9) / (time/1000.0);
}

std::vector<float> read_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open file");
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<float> buffer(size / sizeof(float));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) throw std::runtime_error("failed to read file");
  return buffer;
}

bool validate_avx2(const float *gpu, const float *ref, size_t n) {
    size_t i = 0;
    const __m256 tolv   = _mm256_set1_ps(TOL);
    const __m256 signbit= _mm256_set1_ps(-0.f);   // bitmask for abs

    for (; i + 8 <= n; i += 8) {
        __m256 vg   = _mm256_loadu_ps(gpu + i);                // load gpuC[i..i+7]
        __m256 vr   = _mm256_loadu_ps(ref + i);                // load staging_c[i..i+7]
        __m256 diff = _mm256_sub_ps(vr, vg);                   // ref - gpu
        __m256 ad   = _mm256_andnot_ps(signbit, diff);         // abs(ref - gpu)
        __m256 ag   = _mm256_andnot_ps(signbit, vg);           // abs(gpu)
        __m256 thr  = _mm256_mul_ps(tolv, ag);                 // tol * abs(gpu)
        __m256 cmp  = _mm256_cmp_ps(ad, thr, _CMP_GT_OS);      // ad > thr
        int mask    = _mm256_movemask_ps(cmp);                 // 8-bit mask
        if (mask) {
            int offset = __builtin_ctz(mask);                  // index of first 1-bit
            size_t idx = i + offset;
            printf("expected %f got %f at %zu\n", ref[idx], gpu[idx], idx);
            return false;
        }
    }
    for (; i < n; ++i) {
        float a = fabsf(gpu[i] - ref[i]);
        if (a > TOL * fabsf(gpu[i])) {
            printf("expected %f got %f at %zu\n", ref[i], gpu[i], i);
            return false;
        }
    }
    return true;
}

bool validate_avx512(const float *gpu, const float *ref, size_t n) {
    size_t i = 0;
    const __m512 tolv    = _mm512_set1_ps(TOL);
    const __m512 signbit = _mm512_set1_ps(-0.f);

    for (; i + 16 <= n; i += 16) {
        __m512 vg   = _mm512_loadu_ps(gpu + i);
        __m512 vr   = _mm512_loadu_ps(ref + i);
        __m512 ad   = _mm512_abs_ps(_mm512_sub_ps(vr, vg));
        __m512 thr  = _mm512_mul_ps(tolv, _mm512_abs_ps(vg));
        __mmask16 m= _mm512_cmp_ps_mask(ad, thr, _CMP_GT_OS);
        if (m) {
            int offset= _tzcnt_u32((unsigned) m);
            size_t idx = i + offset;
            printf("expected %f got %f at %zu\n", ref[idx], gpu[idx], idx);
            return false;
        }
    }

    for (; i < n; ++i) {
      float a = fabsf(gpu[i] - ref[i]);
      if (a > TOL * fabsf(gpu[i])) {
          printf("expected %f got %f at %zu\n", ref[i], gpu[i], i);
          return false;
      }
    }
    return true;
}
  
bool cpu_val(buffers bufs) {
  std::vector<float> gpuC(bufs.staging_c.size()); 
  cudaMemcpy(gpuC.data(), bufs.C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  bool ok;
  size_t count = bufs.staging_c.size();


  #if defined(__AVX512F__)
    ok = validate_avx512(gpuC.data(), bufs.staging_c.data(), count);
  #elif defined(__AVX2__)
    ok = validate_avx2(gpuC.data(), bufs.staging_c.data(), count);
  #endif
    return ok;
}

buffers allocs() {
 buffers bufs;
 const unsigned long sz = N * N * sizeof(float);
 cudaMalloc(&bufs.A, sz);
 cudaMalloc(&bufs.B, sz);
 cudaMalloc(&bufs.C, sz);
 cudaMalloc(&bufs.B_t, sz);
 auto a = read_binary("./bins/A.bin");
 auto b = read_binary("./bins/B.bin");
 auto b_t = read_binary("./bins/B_t.bin");
 bufs.staging_c = read_binary("./bins/C.bin");
 cudaMemcpy(bufs.A, a.data(), sz, cudaMemcpyHostToDevice);
 cudaMemcpy(bufs.B, b.data(), sz, cudaMemcpyHostToDevice);
 cudaMemcpy(bufs.B_t, b_t.data(), sz, cudaMemcpyHostToDevice);
 return bufs;
}