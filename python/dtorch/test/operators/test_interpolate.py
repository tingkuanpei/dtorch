"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_interpolate(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.nn.functional.interpolate(torch_in, size=(16, 16))
    dtorch_out = dtorch.nn.functional.interpolate(dtorch_in, size=(16, 16))
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestInterpolate(unittest.TestCase):
    def test_interpolate(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 4, 8, 8)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_interpolate(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
