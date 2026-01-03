#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 <kernel.cu>"
  echo "Generates PTX and SASS assembly from a CUDA kernel file"
  exit 1
fi

KERNEL_FILE="$1"
BASENAME=$(basename "$KERNEL_FILE" .cu)

if [ ! -f "$KERNEL_FILE" ]; then
  echo "Error: File '$KERNEL_FILE' not found"
  exit 1
fi

# Auto-detect GPU architecture
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
if [ -z "$GPU_ARCH" ]; then
  echo "Warning: Could not detect GPU, defaulting to sm_75 (Turing)"
  GPU_ARCH=75
fi

ARCH="-gencode=arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}"

# Generate PTX
echo "Generating PTX to ${BASENAME}.ptx..."
nvcc $ARCH -O3 -use_fast_math --ptx "$KERNEL_FILE" -o "${BASENAME}.ptx"

# Generate SASS (cubin)
echo "Generating SASS to ${BASENAME}.sass..."
nvcc $ARCH -O3 -use_fast_math -cubin "$KERNEL_FILE" -o "${BASENAME}.cubin"
cuobjdump -sass "${BASENAME}.cubin" > "${BASENAME}.sass"
rm "${BASENAME}.cubin"

echo "Done! Generated:"
echo "  - ${BASENAME}.ptx"
echo "  - ${BASENAME}.sass"
