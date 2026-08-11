"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.util.benchmark import Benchmark
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal
from dtorch.test.system_overhead.util import print_benchmark_result


def _benchmark_tensor_add(test_case, device):
    warmup = 100
    count = 500

    shapes = [
        (1,),
        (1, 1, 1024),
        (1, 256, 1024),
        (1, 512, 1024),
        (1, 1024, 1024),
        (2, 1024, 1024),
        (4, 1024, 1024),
        (8, 1024, 1024),
        (16, 1024, 1024),
        (32, 1024, 1024),
        (32, 1024, 1024),
        (64, 1024, 1024),
        (128, 1024, 1024),
        (256, 1024, 1024),
        (512, 1024, 1024),
        (1024, 1024, 1024),
    ]

    if device == "cpu":
        shapes = shapes[:8]

    torch_metrics = []
    dtorch_metrics = []

    for shape in shapes:
        torch_a = torch.rand(*shape, dtype=torch.float16, device=device)
        torch_b = torch.rand(*shape, dtype=torch.float16, device=device)

        @torch.inference_mode()
        def torch_func():
            return torch_a + torch_b

        torch_metric, torch_out = Benchmark.run(torch_func, warmup=warmup, count=count)

        dtorch_a = dtorch.Tensor(torch_a)
        dtorch_b = dtorch.Tensor(torch_b)

        def dtorch_func():
            return dtorch_a + dtorch_b

        dtorch_metric, dtorch_out = Benchmark.run(dtorch_func, warmup=warmup, count=count)
        assert_tensor_equal(test_case, torch_out, dtorch_out)

        torch_metrics.append(torch_metric)
        dtorch_metrics.append(dtorch_metric)

    rows = [
        (shape, torch_m.duration, dtorch_m.duration)
        for shape, dtorch_m, torch_m in zip(shapes, dtorch_metrics, torch_metrics)
    ]
    print_benchmark_result("tensor_add", device, warmup, count, rows)


class TestTensorAdd(unittest.TestCase):
    def test_benchmark_tensor_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _benchmark_tensor_add(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
