"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""


def print_benchmark_result(name, device, warmup, count, rows):
    """Print benchmark results in a formatted table.

    Args:
        name: benchmark name (e.g. "tensor_add", "sd3_transformer")
        device: device string ("cpu" or "cuda")
        warmup: warmup iteration count
        count: benchmark iteration count
        rows: list of (shape, torch_duration, dtorch_duration) tuples
    """
    sep = "─" * 80
    print(f"\n{sep}")
    print(f"  Benchmark: {name}  (device={device}, warmup={warmup}, count={count})")
    print(sep)
    print(f"  {'Shape':30s} {'PyTorch/s':>15s} {'DTorch/s':>15s} {'Ratio':>10s}")
    print(f"  {'─' * 30} {'─' * 15} {'─' * 15} {'─' * 10}")
    for shape, torch_duration, dtorch_duration in rows:
        ratio = dtorch_duration / torch_duration
        shape_str = str(shape)
        print(f"  {shape_str:30s} {torch_duration:>15.4e} {dtorch_duration:>15.4e} {ratio:>10.4f}")
    print(sep)
