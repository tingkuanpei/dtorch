"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_squeeze(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.squeeze(torch_in, dim=(1))
    dtorch_out = dtorch.squeeze(dtorch_in, dim=(1))
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.squeeze(torch_in)
    dtorch_out = dtorch.squeeze(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch_in.squeeze(1)
    dtorch_out = dtorch_in.squeeze(1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestSqueeze(unittest.TestCase):
    def test_squeeze(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 1, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_squeeze(test_case, *arg)


def _test_unsqueeze(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.unsqueeze(torch_in, dim=1)
    dtorch_out = dtorch.unsqueeze(dtorch_in, dim=1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.unsqueeze(torch_in, dim=0)
    dtorch_out = dtorch.unsqueeze(dtorch_in, dim=0)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.unsqueeze(torch_in, dim=3)
    dtorch_out = dtorch.unsqueeze(dtorch_in, dim=3)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.unsqueeze(torch_in, dim=-1)
    dtorch_out = dtorch.unsqueeze(dtorch_in, dim=-1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.unsqueeze(torch_in, dim=-4)
    dtorch_out = dtorch.unsqueeze(dtorch_in, dim=-4)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestUnsqueeze(unittest.TestCase):
    def test_unsqueeze(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_unsqueeze(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
