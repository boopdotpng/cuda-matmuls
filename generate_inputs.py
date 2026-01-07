#!/usr/bin/env python3

import numpy as np
from pathlib import Path


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


def generate_matmul_data_f16(n: int, out_dir: Path) -> None:
  """Generate fp16 inputs with fp32 accumulation using numpy."""
  out_dir.mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(seed=0)
  a_np = rng.random((n, n)).astype(np.float32)
  b_np = rng.random((n, n)).astype(np.float32)

  a_fp16 = a_np.astype(np.float16)
  b_fp16 = b_np.astype(np.float16)
  c_fp32 = a_fp16.astype(np.float32) @ b_fp16.astype(np.float32)

  a_fp16.tofile(out_dir / "A.bin")
  b_fp16.tofile(out_dir / "B.bin")
  b_fp16.T.tofile(out_dir / "B_t.bin")
  c_fp32.astype(np.float16).tofile(out_dir / "C.bin")

  print(f"saved fp16 A * B (both {n}x{n}) and fp16 C to {out_dir}/")


def main() -> None:
  n = 4096

  # Generate fp32 data
  generate_matmul_data_numpy(n, Path("data/f32"))

  # Generate fp16 data using numpy
  generate_matmul_data_f16(n, Path("data/f16"))


if __name__ == "__main__":
  main()
