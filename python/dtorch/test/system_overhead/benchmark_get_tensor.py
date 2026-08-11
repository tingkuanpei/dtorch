"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict
import time

import torch

import dtorch
from dtorch.util.benchmark import Benchmark
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal
from dtorch.test.system_overhead.util import print_benchmark_result


def _benchmark_get_tensor(test_case, device):
    """Benchmark the overhead of converting a DTorch tensor to a PyTorch tensor via to_torch()."""
    warmup = 10
    count = 100

    shapes = [
        (1,),
        (256, 256),
        (1024, 1024),
        (4096, 1024),
        (8192, 4096),
    ]

    torch_metrics = []
    dtorch_metrics = []

    for shape in shapes:
        torch_a = torch.rand(1, dtype=torch.float16, device=device)
        torch_b = torch.rand(1, dtype=torch.float16, device=device)
        dtorch_a = dtorch.Tensor(torch_a)
        dtorch_b = dtorch.Tensor(torch_b)

        def torch_func():
            return torch_a + torch_b

        def dtorch_func():
            return (dtorch_a + dtorch_b).to_torch()

        torch_metric, torch_out = Benchmark.run(torch_func, warmup=warmup, count=count)
        dtorch_metric, dtorch_out = Benchmark.run(dtorch_func, warmup=warmup, count=count)
        torch_metrics.append(torch_metric)
        dtorch_metrics.append(dtorch_metric)

        assert_tensor_equal(test_case, torch_out, dtorch_out)

    rows = [
        (shape, torch_m.duration, dtorch_m.duration)
        for shape, dtorch_m, torch_m in zip(shapes, dtorch_metrics, torch_metrics)
    ]
    print_benchmark_result("get_tensor", device, warmup, count, rows)


class TestGetTensor(unittest.TestCase):
    def test_benchmark_get_tensor(self):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _benchmark_get_tensor(self, *arg)


if __name__ == "__main__":
    unittest.main()
