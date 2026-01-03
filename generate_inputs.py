#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from tinygrad import Tensor, dtypes


def generate_matmul_data_numpy(n: int, out_dir: Path) -> None:
  """Generate fp32 matrix multiplication data using numpy."""
  out_dir.mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(seed=0)
  a = rng.random((n, n)).astype(np.float32)
  b = rng.random((n, n)).astype(np.float32)

  a.tofile(out_dir / "A.bin")
  b.tofile(out_dir / "B.bin")
  b.T.tofile(out_dir / "B_t.bin")
  (a @ b).tofile(out_dir / "C.bin")

  print(f"saved fp32 A * B (both {n}x{n}) and C to {out_dir}/")


def generate_matmul_data_tinygrad(n: int, out_dir: Path) -> None:
  """Generate fp16 matrix multiplication data using tinygrad on GPU."""
  out_dir.mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(seed=0)
  a_np = rng.random((n, n)).astype(np.float32)
  b_np = rng.random((n, n)).astype(np.float32)

  # Convert to tinygrad tensors and cast to fp16
  a = Tensor(a_np, dtype=dtypes.float16)
  b = Tensor(b_np, dtype=dtypes.float16)

  # Compute matmul on GPU
  c = a @ b

  # Convert back to numpy
  a_fp16 = a.numpy().astype(np.float16)
  b_fp16 = b.numpy().astype(np.float16)
  c_fp16 = c.numpy().astype(np.float16)

  a_fp16.tofile(out_dir / "A.bin")
  b_fp16.tofile(out_dir / "B.bin")
  b_fp16.T.tofile(out_dir / "B_t.bin")
  c_fp16.tofile(out_dir / "C.bin")

  print(f"saved fp16 A * B (both {n}x{n}) and C to {out_dir}/")


def main() -> None:
  n = 4096

  # Generate fp32 data
  generate_matmul_data_numpy(n, Path("data/f32"))

  # Generate fp16 data using tinygrad
  generate_matmul_data_tinygrad(n, Path("data/f16"))


if __name__ == "__main__":
  main()
