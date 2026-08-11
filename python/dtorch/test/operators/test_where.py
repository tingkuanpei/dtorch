"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_where(test_case, shape, device):
    torch_a = torch.rand(*shape, dtype=torch.float32, device=device)
    torch_b = torch.rand(*shape[1:], dtype=torch.float32, device=device)
    dtorch_a = dtorch.Tensor(torch_a)
    dtorch_b = dtorch.Tensor(torch_b)

    torch_out = torch.where(torch_a > 0, torch_a, torch_b)
    dtorch_out = dtorch.where(dtorch_a > 0, dtorch_a, dtorch_b)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.where(torch_a > 0, 1.0, torch_b)
    dtorch_out = dtorch.where(dtorch_a > 0, 1.0, dtorch_b)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestWhere(unittest.TestCase):
    def test_where(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_where(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
