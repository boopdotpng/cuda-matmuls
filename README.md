# Increasingly fast CUDA matmuls

CUDA matmul kernels that get progressively faster.

## Structure

- `f32/`: fp32 CUDA kernels (naive, tiled)
- `f16/`: fp16 CUDA kernels (to be implemented)
- `utils.h`, `utils.cu`: shared utilities and benchmark framework
- `data/`: generated input/output tensors (not committed)
- `generate_inputs.py`: generate test data (uses tinygrad for fp16)
- `ncu_aggregate.py`: aggregate NCU profiler outputs
- `cu2asm.sh`: convert .cu files to PTX/SASS assembly

## Quickstart

Python deps: `numpy`, `tinygrad` (for fp16 generation)

Generate input tensors:
```bash
python3 generate_inputs.py
```

Build (auto-detects GPU architecture):
```bash
make
```

Run:
```bash
./f32_naive
./f32_tiled
```

Generate assembly:
```bash
./cu2asm.sh f32/tiled.cu
```
