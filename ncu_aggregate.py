#!/usr/bin/env python3
import sys
import re
from tabulate import tabulate

def parse_runs(lines):
    runs = []
    current = None
    start = False
    for line in lines:
        if line.strip().startswith("==PROF== Disconnected"):
            start = True
            continue
        if not start:
            continue
        indent = len(line) - len(line.lstrip(' '))
        if indent == 2 and '(' in line:
            if current:
                runs.append(current)
            current = {}
            continue
        if current is None:
            continue
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        unit = parts[-2]
        name = ' '.join(parts[:-2])
        current[name] = (value, unit)
    if current:
        runs.append(current)
    return runs


def aggregate(runs):
    if len(runs) <= 1:
        sys.exit("not enough runs to aggregate after discarding first.")
    data = runs[1:]
    n = len(data)
    sums = {}
    units = {}
    for run in data:
        for name, (value, unit) in run.items():
            sums[name] = sums.get(name, 0.0) + value
            units.setdefault(name, unit)
    avgs = {name: sums[name] / n for name in sums}
    return avgs, units, n


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ncu_output.txt>")
        sys.exit(1)
    fname = sys.argv[1]
    with open(fname) as f:
        lines = f.readlines()
    runs = parse_runs(lines)
    avgs, units, count = aggregate(runs)
    rows = [(name, f"{avgs[name]:.2f}", units[name]) for name in sorted(avgs)]
    print(f"aggregated metrics over {count} runs (first run discarded):\n")
    print(tabulate(rows, headers=["Metric", "Average", "Unit"], tablefmt="github"))

if __name__ == '__main__':
    main()
