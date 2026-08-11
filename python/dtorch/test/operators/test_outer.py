"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_outer(test_case, m, n, device):
    torch_a = torch.rand(m, dtype=torch.float32, device=device)
    torch_b = torch.rand(n, dtype=torch.float32, device=device)
    dtorch_a = dtorch.Tensor(torch_a)
    dtorch_b = dtorch.Tensor(torch_b)

    torch_out = torch.outer(torch_a, torch_b)
    dtorch_out = dtorch.outer(dtorch_a, dtorch_b)
    assert_tensor_equal(test_case, torch_out, dtorch_out)
    dtorch_out = dtorch_a.outer(dtorch_b)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_b = torch.rand(n, dtype=torch.float64, device=device)
    dtorch_b = dtorch.Tensor(torch_b)
    torch_out = torch.outer(torch_a, torch_b)
    dtorch_out = dtorch.outer(dtorch_a, dtorch_b)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestOuter(unittest.TestCase):
    def test_outer(test_case):
        arg_dict = OrderedDict()
        arg_dict["m"] = [2, 4, 5]
        arg_dict["n"] = [2, 4, 5]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_outer(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
