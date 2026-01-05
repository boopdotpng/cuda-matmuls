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

UTILS = utils.cu

all: f32_naive f32_tiled f16_wmma

f32_naive: f32/naive.cu $(UTILS)
	$(NVCC) $(NVCCFLAGS) f32/naive.cu $(UTILS) -o f32_naive

f32_tiled: f32/tiled.cu $(UTILS)
	$(NVCC) $(NVCCFLAGS) f32/tiled.cu $(UTILS) -o f32_tiled

f16_wmma: f16/wmma.cu $(UTILS)
	$(NVCC) $(NVCCFLAGS) f16/wmma.cu $(UTILS) -o f16_wmma

clean:
	rm -f f32_naive f32_tiled f16_wmma

.PHONY: all clean
