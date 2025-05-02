#!/usr/bin/env python3
import sys
import re
from tabulate import tabulate

def parse_runs(lines):
    runs = []
    current_metrics = None
    current_kernel = None
    started = False
    for line in lines:
        if line.strip().startswith("==PROF== Disconnected"):
            started = True
            continue
        if not started:
            continue
        indent = len(line) - len(line.lstrip(' '))
        # detect new kernel block
        if indent == 2 and '(' in line:
            kernel_name = line.strip().split('(')[0]
            if current_metrics:
                runs.append((current_kernel, current_metrics))
            current_kernel = kernel_name
            current_metrics = {}
            continue
        if current_metrics is None:
            continue
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        unit = parts[-2]
        metric = ' '.join(parts[:-2])
        current_metrics[metric] = (value, unit)
    if current_metrics:
        runs.append((current_kernel, current_metrics))
    return runs


def aggregate_runs(runs):
    from collections import defaultdict
    grouped = defaultdict(list)
    for kernel, metrics in runs:
        grouped[kernel].append(metrics)
    avg_data = {}
    units = {}
    for kernel, metrics_list in grouped.items():
        if len(metrics_list) <= 1:
            continue
        data = metrics_list[1:]  # discard first
        n = len(data)
        sums = {}
        for m in data:
            for name, (value, unit) in m.items():
                sums[name] = sums.get(name, 0.0) + value
                units[name] = unit
        avg_data[kernel] = {name: sums[name] / n for name in sums}
    return avg_data, units


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <ncu_file1> [ncu_file2 ...]")
        sys.exit(1)
    file_results = []
    for fname in sys.argv[1:]:
        with open(fname) as f:
            lines = f.readlines()
        runs = parse_runs(lines)
        avg_by_kernel, units = aggregate_runs(runs)
        # assume one kernel per file
        if not avg_by_kernel:
            print(f"no aggregate data for {fname}")
            continue
        kernel = next(iter(avg_by_kernel))
        file_results.append((kernel, avg_by_kernel[kernel]))
    # collect all metrics
    all_metrics = set()
    for _, avg in file_results:
        all_metrics.update(avg.keys())
    rows = []
    for metric in sorted(all_metrics):
        row = [metric]
        for kernel, avg in file_results:
            row.append(f"{avg.get(metric, 0.0):.2f}")
        row.append(units.get(metric, ''))
        rows.append(row)
    headers = ['Metric'] + [f"Average ({k})" for k, _ in file_results] + ['Unit']
    print(tabulate(rows, headers=headers, tablefmt='github'))

if __name__ == '__main__':
    main()
