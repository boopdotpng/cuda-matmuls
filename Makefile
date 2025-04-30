# 5070 ti 
ARCH        = -gencode=arch=compute_120,code=sm_120
NVCC        = nvcc
NVCCFLAGS   = $(ARCH) -I utils/ -Wno-deprecated-gpu-targets -O3 -use_fast_math \
              -Xcompiler "-O3 -mavx2 -mfma -march=native"

UTILS       = utils/utils.cu
NAIVE_SRC   = naive.cu
TILED_SRC   = tiled.cu
ULTRAFAST_SRC = ultrafast.cu
CPU_SRC     = cpu_matmul.cpp

NAIVE_BIN   = outs/naive
TILED_BIN   = outs/tiled
ULTRAFAST_BIN = outs/ultrafast
CPU_BIN     = outs/cpu_matmul

GCC         = g++
GCCFLAGS    = -march=native -fopenmp -O2

all: naive tiled ultrafast cpu

naive: $(NAIVE_SRC) $(UTILS)
	@mkdir -p outs
	$(NVCC) $(NVCCFLAGS) $(NAIVE_SRC) $(UTILS) -o $(NAIVE_BIN)

tiled: $(TILED_SRC) $(UTILS)
	@mkdir -p outs
	$(NVCC) $(NVCCFLAGS) $(TILED_SRC) $(UTILS) -o $(TILED_BIN)

ultrafast: $(ULTRAFAST_SRC) $(UTILS)
	@mkdir -p outs
	$(NVCC) $(NVCCFLAGS) $(ULTRAFAST_SRC) $(UTILS) -o $(ULTRAFAST_BIN)

cpu: $(CPU_SRC)
	@mkdir -p outs
	$(GCC) $(GCCFLAGS) $(CPU_SRC) -o $(CPU_BIN)

clean:
	rm -f $(NAIVE_BIN) $(TILED_BIN) $(ULTRAFAST_BIN) $(CPU_BIN)
