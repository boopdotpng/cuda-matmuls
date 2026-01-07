#!/usr/bin/env python3
import argparse
import csv
import datetime
import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import List, Tuple, Optional

DEFAULT_WARMUP_RUNS = 10
DEFAULT_BENCH_RUNS = 5
DEFAULT_LAST_LAUNCH_SKIP = DEFAULT_WARMUP_RUNS + DEFAULT_BENCH_RUNS - 1

KERNEL_RE = re.compile(
  r"__global__\s+void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(",
  re.MULTILINE
)

def which(cmd: str) -> Optional[str]:
  return shutil.which(cmd)

def now_str() -> str:
  return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_tee(cmd: List[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> Tuple[int, str, str]:
  """
  Runs a command quietly and returns (returncode, stdout, stderr) as strings.
  """
  p = subprocess.Popen(
    cmd,
    cwd=cwd,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
  )
  out, err = p.communicate()
  return p.returncode, out or "", err or ""

def write_header_md(f, title: str):
  f.write("\n")
  f.write(f"## {title}\n\n")

def write_cmd_block_md(f, label: str, cmd: List[str], rc: int, out: str, err: str, out_is_md: bool = False):
  write_header_md(f, label)
  f.write("Command:\n\n")
  f.write("```bash\n")
  f.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
  f.write("```\n\n")
  f.write(f"Return code: `{rc}`\n\n")
  if out.strip():
    f.write("Output:\n\n")
    if out_is_md:
      f.write(out)
      if not out.endswith("\n"):
        f.write("\n")
    else:
      f.write("```text\n")
      f.write(out)
      if not out.endswith("\n"):
        f.write("\n")
      f.write("```\n")
  if err.strip():
    f.write("\nStderr:\n\n```text\n")
    f.write(err)
    if not err.endswith("\n"):
      f.write("\n")
    f.write("```\n")

def extract_cuobjdump_function_block(text: str, kernel_substr: str) -> str:
  """
  Extracts cuobjdump blocks like:
    Function : <name>
    ...
  For both SASS and PTX, cuobjdump often prints function headers.
  We match by substring to handle mangling/demangling differences.
  """
  lines = text.splitlines(True)
  out = []
  i = 0
  found_any = False
  blocks = []

  func_re = re.compile(r"^\s*Function\s*:\s*(.+)\s*$")
  entry_re = re.compile(r"^\s*\.entry\s+([^\s(]+).*")

  while i < len(lines):
    m = func_re.match(lines[i])
    if m:
      func_name = m.group(1)
      block = [lines[i]]
      i += 1
      while i < len(lines) and not func_re.match(lines[i]):
        block.append(lines[i])
        i += 1
      blocks.append((func_name, "".join(block)))
      if kernel_substr in func_name:
        out.extend(block)
        found_any = True
      continue

    # PTX dumps may not have "Function :" but do have ".entry"
    m2 = entry_re.match(lines[i])
    if m2:
      entry_name = m2.group(1)
      block = [lines[i]]
      i += 1
      while i < len(lines) and not entry_re.match(lines[i]):
        block.append(lines[i])
        i += 1
      blocks.append((entry_name, "".join(block)))
      if kernel_substr in entry_name or kernel_substr in "".join(block):
        out.extend(block)
        found_any = True
      continue

    i += 1

  if found_any:
    return "".join(out)

  if not blocks:
    head = "".join(lines[:200])
    return "(No cuobjdump Function or .entry blocks found; this dump likely lacks PTX/SASS for the target.)\n\n[cuobjdump head]\n" + head

  if len(blocks) == 1:
    return "(Only one cuobjdump Function/.entry block found; showing it.)\n\n" + blocks[0][1]

  # fallback: return a small hint + first ~200 lines so you can see what cuobjdump produced
  head = "".join(lines[:200])
  sample = ", ".join(name for name, _ in blocks[:5])
  return (
    f"(No cuobjdump function/entry block matched substring: {kernel_substr!r}. "
    f"Sample blocks: {sample})\n\n[cuobjdump head]\n{head}"
  )

def resolve_launch_selection(skip: Optional[int], count: Optional[int]) -> Tuple[int, int, bool]:
  if skip is None and count is None:
    return DEFAULT_LAST_LAUNCH_SKIP, 1, True
  if skip is None:
    return 0, count if count is not None else 1, False
  return skip, count if count is not None else 1, False

def extract_first_kernel_from_source(path: str) -> Optional[str]:
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = f.read()
  except OSError:
    return None
  m = KERNEL_RE.search(data)
  if not m:
    return None
  return m.group(1)

def source_from_makefile(exe_base: str, cwd: str) -> Optional[str]:
  makefile = os.path.join(cwd, "Makefile")
  if not os.path.exists(makefile):
    return None
  try:
    with open(makefile, "r", encoding="utf-8") as f:
      lines = f.readlines()
  except OSError:
    return None
  for line in lines:
    if ":" not in line:
      continue
    target, deps = line.split(":", 1)
    if target.strip() != exe_base:
      continue
    for token in deps.split():
      if token.endswith(".cu"):
        return os.path.join(cwd, token)
  return None

def source_from_exe_name(exe_base: str, cwd: str) -> Optional[str]:
  if "_" not in exe_base:
    return None
  prefix, rest = exe_base.split("_", 1)
  candidate = os.path.join(cwd, prefix, f"{rest}.cu")
  if os.path.exists(candidate):
    return candidate
  return None

def csv_blocks_from_text(text: str) -> Tuple[List[List[List[str]]], List[str]]:
  blocks = []
  notes = []
  current = []
  for line in text.splitlines():
    try:
      row = next(csv.reader([line]))
    except csv.Error:
      row = []
    if len(row) > 1:
      current.append(row)
      continue
    if current:
      blocks.append(current)
      current = []
    if line.strip():
      notes.append(line)
  if current:
    blocks.append(current)
  return blocks, notes

def csv_block_to_markdown(block: List[List[str]]) -> str:
  if not block:
    return ""
  header = block[0]
  data = block[1:]
  def esc(cell: str) -> str:
    return cell.replace("|", "\\|")
  out = []
  out.append("| " + " | ".join(esc(c) for c in header) + " |\n")
  out.append("| " + " | ".join("---" for _ in header) + " |\n")
  for row in data:
    out.append("| " + " | ".join(esc(c) for c in row) + " |\n")
  return "".join(out)

def csv_text_to_markdown(text: str) -> str:
  blocks, notes = csv_blocks_from_text(text)
  out = []
  if notes:
    out.append("Notes:\n\n```text\n")
    out.append("\n".join(notes))
    out.append("\n```\n\n")
  for block in blocks:
    table = csv_block_to_markdown(block)
    if table:
      out.append(table)
      out.append("\n")
  return "".join(out).rstrip() + "\n"

def extract_sass_instructions(sass_dump: str) -> str:
  """Extract just the instruction mnemonics and operands from SASS dump."""
  lines = sass_dump.splitlines()
  instructions = []
  inst_re = re.compile(r'^\s*/\*[0-9a-f]+\*/\s+(.+?)\s*(?:;.*)?$', re.IGNORECASE)

  for line in lines:
    m = inst_re.match(line)
    if m:
      instructions.append(m.group(1).strip())

  return "\n".join(instructions) if instructions else "(No SASS instructions found)"

def main():
  ap = argparse.ArgumentParser(
    description="Run an executable normally, then run curated Nsight Compute passes + dump PTX/SASS into one report."
  )
  ap.add_argument("exe", help="Path to the compiled CUDA executable to run (e.g. ./f16_wmma)")
  ap.add_argument("kernel", nargs="?", default=None, help="Kernel name substring (omit to auto-detect from the source). Used for filtering + ncu -k regex:...")
  ap.add_argument("--src", default=None, help="CUDA source file to auto-detect the kernel from (default: inferred from Makefile/exe name)")
  ap.add_argument("--skip", type=int, default=None, help=f"ncu launch skip (default: last launch; skip {DEFAULT_LAST_LAUNCH_SKIP})")
  ap.add_argument("--count", type=int, default=None, help="ncu launch count (default: 1)")
  ap.add_argument("--out", default=None, help="Output report path (default: <exe>_<kernel>_profile.md)")
  ap.add_argument("--cwd", default=None, help="Working directory (default: current)")
  ap.add_argument("--", dest="exe_args", nargs=argparse.REMAINDER, help="Arguments passed to the executable")
  args = ap.parse_args()

  exe = args.exe
  kernel = args.kernel
  exe_args = args.exe_args if args.exe_args is not None else []
  launch_skip, launch_count, auto_last = resolve_launch_selection(args.skip, args.count)

  if not os.path.exists(exe):
    print(f"ERROR: executable not found: {exe}", file=sys.stderr)
    sys.exit(1)

  kernel_source = None
  kernel_auto = False
  if kernel is None:
    cwd = args.cwd or os.getcwd()
    if args.src:
      kernel_source = args.src
      if not os.path.isabs(kernel_source):
        kernel_source = os.path.join(cwd, kernel_source)
      if not os.path.exists(kernel_source):
        print(f"ERROR: source file not found: {kernel_source}", file=sys.stderr)
        sys.exit(1)
    else:
      exe_base = os.path.basename(exe)
      kernel_source = source_from_makefile(exe_base, cwd) or source_from_exe_name(exe_base, cwd)
    if kernel_source:
      kernel = extract_first_kernel_from_source(kernel_source)
    if not kernel:
      print("WARNING: failed to auto-detect kernel name. Profiling all kernels.", file=sys.stderr)
      kernel = ".*"
    kernel_auto = True

  exe_base = os.path.basename(exe)
  safe_kernel = re.sub(r"[^A-Za-z0-9_.-]+", "_", kernel)
  out_path = args.out or f"{exe_base}_{safe_kernel}_profile.md"

  tools = {
    "ncu": which("ncu"),
    "nsys": which("nsys"),
    "cuobjdump": which("cuobjdump"),
    "nvdisasm": which("nvdisasm"),
    "nvcc": which("nvcc"),
    "nvidia-smi": which("nvidia-smi"),
    "uname": which("uname"),
    "lsb_release": which("lsb_release"),
  }

  nsys_report = None
  with open(out_path, "w", encoding="utf-8") as f:
    f.write("# CUDA Profile Report\n")

    # 1) Normal run
    cmd_normal = [exe] + exe_args
    rc, out, err = run_tee(cmd_normal, cwd=args.cwd)
    write_cmd_block_md(f, "STEP 1: NORMAL RUN (no profiling)", cmd_normal, rc, out, err)

    # 2) PTX / SASS dumps (best-effort via cuobjdump)
    if tools["cuobjdump"]:
      cmd_ptx = [tools["cuobjdump"], "--dump-ptx", exe]
      rc, out, err = run_tee(cmd_ptx, cwd=args.cwd)
      filtered = extract_cuobjdump_function_block(out, kernel)
      write_cmd_block_md(f, "STEP 2A: cuobjdump --dump-ptx (filtered to kernel)", cmd_ptx, rc, filtered, err)

      cmd_sass = [tools["cuobjdump"], "--dump-sass", exe]
      rc, out, err = run_tee(cmd_sass, cwd=args.cwd)
      filtered = extract_cuobjdump_function_block(out, kernel)
      write_cmd_block_md(f, "STEP 2B: cuobjdump --dump-sass (filtered to kernel)", cmd_sass, rc, filtered, err)

      # Extract just the instructions in a simple format
      sass_instructions = extract_sass_instructions(filtered)
      write_header_md(f, "STEP 2C: SASS Instructions (assembly format)")
      f.write("```asm\n")
      f.write(sass_instructions)
      if not sass_instructions.endswith("\n"):
        f.write("\n")
      f.write("```\n")
    else:
      write_header_md(f, "STEP 2: cuobjdump not found")
      f.write("Install CUDA toolkit (or ensure cuobjdump is in PATH) to dump embedded PTX/SASS from the executable.\n")

    # If ncu isn't present, we're done
    if not tools["ncu"]:
      write_header_md(f, "STEP 3: ncu not found")
      f.write("Install Nsight Compute (ncu) or add it to PATH.\n")
      print(f"\nWrote report to: {out_path}")
      return

    # Helper: common ncu args (keep output small, 1 kernel launch)
    ncu_common = [
      tools["ncu"],
      "-k", f"regex:{kernel}",
      "--launch-skip", str(launch_skip),
      "--launch-count", str(launch_count),
    ]

    # 3) Curated profiling: SpeedOfLight (trim to a few sections)
    cmd_sol = ncu_common + ["--csv", "--set", "speedOfLight", exe] + exe_args
    rc, out, err = run_tee(cmd_sol, cwd=args.cwd)
    sol_md = csv_text_to_markdown(out)
    write_cmd_block_md(f, "STEP 3A: ncu --set speedOfLight (csv)", cmd_sol, rc, sol_md, err, out_is_md=True)

    # 4) Stall + scheduler breakdown (usually the most actionable “why am I slow?”)
    cmd_stalls = ncu_common + ["--csv", "--section", "WarpStateStats", "--section", "SchedulerStats", exe] + exe_args
    rc, out, err = run_tee(cmd_stalls, cwd=args.cwd)
    stalls_md = csv_text_to_markdown(out)
    write_cmd_block_md(f, "STEP 3B: ncu WarpStateStats + SchedulerStats", cmd_stalls, rc, stalls_md, err, out_is_md=True)

    # 5) PTX source view via ncu (if available)
    cmd_ptx_src = ncu_common + [
      "--csv",
      "--section", "SourceCounters",
      "--page", "source",
      "--print-source", "ptx",
      exe
    ] + exe_args
    rc, out, err = run_tee(cmd_ptx_src, cwd=args.cwd)
    ptx_md = csv_text_to_markdown(out)
    write_cmd_block_md(f, "STEP 3C: ncu SourceCounters + PTX (page source)", cmd_ptx_src, rc, ptx_md, err, out_is_md=True)

    # 6) SASS source view via ncu (if available)
    cmd_sass_src = ncu_common + [
      "--csv",
      "--section", "SourceCounters",
      "--page", "source",
      "--print-source", "sass",
      exe
    ] + exe_args
    rc, out, err = run_tee(cmd_sass_src, cwd=args.cwd)
    sass_md = csv_text_to_markdown(out)
    write_cmd_block_md(f, "STEP 3D: ncu SourceCounters + SASS (page source)", cmd_sass_src, rc, sass_md, err, out_is_md=True)

    # 7) A very small “did we hit tensor pipe?” check (names can vary; best-effort)
    cmd_list = [tools["ncu"], "--list-metrics"]
    rc_l, out_l, err_l = run_tee(cmd_list, cwd=args.cwd)
    # Find one plausible tensor inst counter if present
    tensor_metric = None
    for line in out_l.splitlines():
      if "inst_executed" in line and ("tensor" in line.lower() or "pipe_tensor" in line.lower()):
        tensor_metric = line.strip().split()[0]
        break
    if tensor_metric:
      cmd_tensor = ncu_common + ["--csv", "--metrics", f"{tensor_metric}", exe] + exe_args
      rc, out, err = run_tee(cmd_tensor, cwd=args.cwd)
      tensor_md = csv_text_to_markdown(out)
      write_cmd_block_md(f, f"STEP 3E: ncu quick tensor counter ({tensor_metric})", cmd_tensor, rc, tensor_md, err, out_is_md=True)
    else:
      write_cmd_block_md(f, "STEP 3E: ncu tensor counter (not auto-detected)", cmd_list, rc_l, out_l[:4000], err_l)

    # 4) Nsight Systems timeline profile (best-effort)
    if tools["nsys"]:
      nsys_out_base = os.path.splitext(out_path)[0] + "_nsys"
      nsys_report = nsys_out_base + ".nsys-rep"
      cmd_nsys = [
        tools["nsys"], "profile",
        "-t", "cuda,nvtx,osrt",
        "--capture-range", "cudaProfilerApi",
        "--capture-range-end", "stop",
        "--force-overwrite", "true",
        "--stats=true",
        "-o", nsys_out_base,
        exe
      ] + exe_args
      rc, out, err = run_tee(cmd_nsys, cwd=args.cwd)
      out_md = csv_text_to_markdown(out) if out.strip() else ""
      note = f"Nsight Systems report base: {nsys_out_base} (expect .nsys-rep)\n"
      if out_md:
        out_md = note + "\n" + out_md
      else:
        out_md = note
      write_cmd_block_md(f, "STEP 4: nsys profile (timeline)", cmd_nsys, rc, out_md, err, out_is_md=True)
    else:
      write_header_md(f, "STEP 4: nsys not found")
      f.write("Install Nsight Systems (nsys) or add it to PATH to capture timeline reports.\n")

  print(f"\nWrote report to: {out_path}")
  if nsys_report:
    print(f"Wrote Nsight Systems report to: {nsys_report}")

if __name__ == "__main__":
  main()
