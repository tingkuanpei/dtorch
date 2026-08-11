"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest

import torch

import dtorch
from dtorch.distributed_spec import init_device_mesh
from dtorch.test.test_util import assert_tensor_allclose


class TestMemory(unittest.TestCase):
    def test_memory_stats(test_case):
        dtorch.default_graph.empty_cache()

        m = 2048
        k = 4096
        n = 1024
        device = "cuda"
        device_mesh = init_device_mesh(
            device,
            [2],
        )
        if not dtorch.default_graph.satisfy(device_mesh):
            return

        torch_in = torch.rand(m, k, dtype=torch.float32, device=device)
        torch_weight = torch.rand(n, k, dtype=torch.float32, device=device)
        torch_out = torch.nn.functional.linear(torch_in, torch_weight)
        dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh)
        dtorch_weight = dtorch.Tensor(torch_weight, device_mesh=device_mesh)
        dtorch_out = dtorch.nn.functional.linear(dtorch_in, dtorch_weight)
        assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)

        dtorch.default_graph.empty_cache()
        memory_stats = dtorch.default_graph.get_memory_stats(
            device_mesh=device_mesh,
            reset_peak=True,
        )
        device_0_memory_stat = memory_stats.find(0)
        test_case.assertTrue(device_0_memory_stat is not None)
        test_case.assertTrue(device_0_memory_stat.memory_allocated > 0)
        test_case.assertTrue(device_0_memory_stat.memory_reserved > 0)
        test_case.assertTrue(device_0_memory_stat.max_memory_allocated > 0)
        test_case.assertTrue(device_0_memory_stat.max_memory_reserved > 0)


if __name__ == "__main__":
    unittest.main()
