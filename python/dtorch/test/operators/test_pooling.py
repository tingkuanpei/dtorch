"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_pooling(test_case, device, dtype):
    torch_in = torch.randn(20, 16, 50, 32, device=device, dtype=dtype)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_m = torch.nn.MaxPool2d(3, stride=2)
    dtorch_m = dtorch.nn.MaxPool2d(3, stride=2)
    torch_out = torch_m(torch_in)
    dtorch_out = dtorch_m(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_m = torch.nn.AdaptiveAvgPool2d((1, 1))
    dtorch_m = dtorch.nn.GlobalAvgPool2d()
    torch_out = torch_m(torch_in)
    dtorch_out = dtorch_m(dtorch_in)
    # assert_tensor_equal(test_case, torch_out, dtorch_out)
    # GlobalAvgPool2d not exactly same as AdaptiveAvgPool2d, so use assert_tensor_allclose to check tensor
    assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)


class TestPooling(unittest.TestCase):
    def test_pooling(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_pooling(
                test_case,
                *arg,
            )


if __name__ == "__main__":
    unittest.main()
