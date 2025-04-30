# 5070ti 
ARCH        = -gencode=arch=compute_120,code=sm_120
NVCC        = nvcc
NVCCFLAGS   = $(ARCH) -I utils/ -Wno-deprecated-gpu-targets -O3 -use_fast_math \
              -Xcompiler "-O3 -mavx2 -mfma -march=native"

UTILS       = utils/utils.cu
NAIVE_SRC   = naive.cu
TILED_SRC   = tiled.cu
ULTRAFAST_SRC = ultrafast.cu

NAIVE_BIN   = outs/naive
TILED_BIN   = outs/tiled
ULTRAFAST_BIN = outs/ultrafast

all: naive tiled ultrafast

naive: $(NAIVE_SRC) $(UTILS)
	@mkdir -p outs
	$(NVCC) $(NVCCFLAGS) $(NAIVE_SRC) $(UTILS) -o $(NAIVE_BIN)

tiled: $(TILED_SRC) $(UTILS)
	@mkdir -p outs
	$(NVCC) $(NVCCFLAGS) $(TILED_SRC) $(UTILS) -o $(TILED_BIN)

ultrafast: $(ULTRAFAST_SRC) $(UTILS)
	@mkdir -p outs
	$(NVCC) $(NVCCFLAGS) $(ULTRAFAST_SRC) $(UTILS) -o $(ULTRAFAST_BIN)

clean:
	rm -f $(NAIVE_BIN) $(TILED_BIN) $(ULTRAFAST_BIN)
