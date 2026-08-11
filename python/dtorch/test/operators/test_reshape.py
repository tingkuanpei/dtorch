"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_reshape(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.reshape(torch_in, shape=(shape[0], -1))
    dtorch_out = dtorch.reshape(dtorch_in, shape=(shape[0], -1))
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestReshape(unittest.TestCase):
    def test_reshape(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_reshape(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
