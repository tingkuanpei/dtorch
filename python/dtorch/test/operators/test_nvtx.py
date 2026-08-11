"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest

import torch

import dtorch
from dtorch.test.test_util import assert_tensor_equal


class TestNvtx(unittest.TestCase):
    def test_functional_use(test_case):
        dtorch.cuda.nvtx.range_push("begin")
        dtorch.cuda.nvtx.range_pop()

        device = "cuda"
        m = 20480
        k = 4096
        n = 1024

        torch_in = torch.rand(m, k, dtype=torch.float32, device=device)
        torch_weight = torch.rand(n, k, dtype=torch.float32, device=device)

        dtorch_in = dtorch.Tensor(torch_in)
        dtorch_weight = dtorch.Tensor(torch_weight)

        torch_out = torch.nn.functional.linear(torch_in, torch_weight)
        dtorch_out = dtorch.nn.functional.linear(dtorch_in, dtorch_weight)
        assert_tensor_equal(test_case, torch_out, dtorch_out)

        for i in range(5):
            with dtorch.cuda.nvtx.range(f"linear{i}"):
                torch_out = torch.nn.functional.linear(torch_in, torch_weight)
                dtorch_out = dtorch.nn.functional.linear(dtorch_in, dtorch_weight)
                assert_tensor_equal(test_case, torch_out, dtorch_out)


if __name__ == "__main__":
    unittest.main()
