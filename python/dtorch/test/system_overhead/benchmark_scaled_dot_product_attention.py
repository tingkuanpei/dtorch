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


def _benchmark_scaled_dot_product_attention(test_case, device):
    warmup = 30
    count = 500

    shapes = [
        (4, 4, 64, 32),
        (4, 8, 128, 64),
        (8, 16, 256, 64),
        (8, 256, 128, 64),
        (8, 1024, 128, 64),
        (8, 4096, 128, 64),
        (8, 8192, 128, 64),
    ]

    torch_metrics = []
    dtorch_metrics = []

    for shape in shapes:
        # torch
        torch_query = torch.rand(*shape, dtype=torch.float16, device=device)
        torch_key = torch.rand(*shape, dtype=torch.float16, device=device)
        torch_value = torch.rand(*shape, dtype=torch.float16, device=device)

        @torch.inference_mode()
        def torch_func():
            return torch.nn.functional.scaled_dot_product_attention(
                torch_query, torch_key, torch_value, dropout_p=0.0, is_causal=False
            )

        torch_metric, torch_out = Benchmark.run(torch_func, warmup=warmup, count=count)

        # dtorch
        dtorch_query = dtorch.Tensor(torch_query)
        dtorch_key = dtorch.Tensor(torch_key)
        dtorch_value = dtorch.Tensor(torch_value)

        def dtorch_func():
            return dtorch.nn.functional.scaled_dot_product_attention(
                dtorch_query, dtorch_key, dtorch_value, dropout_p=0.0, is_causal=False
            )

        dtorch_metric, dtorch_out = Benchmark.run(dtorch_func, warmup=warmup, count=count)
        assert_tensor_equal(test_case, torch_out, dtorch_out)

        torch_metrics.append(torch_metric)
        dtorch_metrics.append(dtorch_metric)

    rows = [
        (shape, torch_m.duration, dtorch_m.duration)
        for shape, dtorch_m, torch_m in zip(shapes, dtorch_metrics, torch_metrics)
    ]
    print_benchmark_result("scaled_dot_product_attention", device, warmup, count, rows)


class TestScaledDotProductAttention(unittest.TestCase):
    def test_benchmark_scaled_dot_product_attention(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        for arg in gen_arg_list(arg_dict):
            _benchmark_scaled_dot_product_attention(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
