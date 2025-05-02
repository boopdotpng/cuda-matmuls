#include <stdio.h>
#include <chrono>
#define N 1024 

float A[N][N], B[N][N], C[N][N];

int main() {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      A[i][j] = B[i][j] = C[i][j] = 1.0f;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      float acc = 0.0;
      for (int k = 0; k < N; k++)
        acc += A[i][k] * B[k][j];
      C[i][j] = acc;
    }
  auto end = std::chrono::high_resolution_clock::now();
  float seconds = std::chrono::duration<float>(end - start).count();
  double flops = 2.0 * N * N * N;
  printf("%.3f GFLOPS\n", (flops / 1e9) / seconds);
  return 0;
}