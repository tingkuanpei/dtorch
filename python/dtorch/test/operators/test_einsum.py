"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_einsum(test_case, device):
    torch_in = torch.rand(4, 5, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.einsum("ij->ji", torch_in)
    dtorch_out = dtorch.einsum("ij->ji", dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestEinsum(unittest.TestCase):
    def test_einsum(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_einsum(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
