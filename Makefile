# Auto-detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
ifeq ($(GPU_ARCH),)
  $(warning Could not detect GPU, defaulting to sm_75 (Turing))
  GPU_ARCH := 75
endif

ARCH        = -gencode=arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) -gencode=arch=compute_$(GPU_ARCH),code=compute_$(GPU_ARCH)
NVCC        = nvcc
NVCCFLAGS   = $(ARCH) -Wno-deprecated-gpu-targets -O3 -use_fast_math \
              -Xcompiler "-O3 -march=native"
CUTLASS_INC = -I./cutlass/include -I./cutlass/tools/util/include

UTILS = utils.cu

all: f32_naive.out f32_tiled.out f16_wmma.out f16_cublas.out

f32_naive.out: f32/naive.cu $(UTILS)
	$(NVCC) $(NVCCFLAGS) f32/naive.cu $(UTILS) -o f32_naive.out

f32_tiled.out: f32/tiled.cu $(UTILS)
	$(NVCC) $(NVCCFLAGS) f32/tiled.cu $(UTILS) -o f32_tiled.out

f16_wmma.out: f16/wmma.cu $(UTILS)
	$(NVCC) $(NVCCFLAGS) f16/wmma.cu $(UTILS) -o f16_wmma.out

f16_cutlass.out: f16/cutlass_gemm.cu $(UTILS)
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_INC) -std=c++17 f16/cutlass_gemm.cu $(UTILS) -o f16_cutlass.out

f16_cublas.out: f16/cublas_gemm.cu $(UTILS)
	$(NVCC) $(NVCCFLAGS) f16/cublas_gemm.cu $(UTILS) -lcublas -o f16_cublas.out

clean:
	rm -f *.out

.PHONY: all clean
