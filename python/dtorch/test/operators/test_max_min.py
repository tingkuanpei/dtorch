"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_max_min(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float16, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.max(torch_in)
    dtorch_out = dtorch.max(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch_in.max()
    dtorch_out = dtorch_in.max()
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out, torch_indice = torch.max(torch_in, dim=1, keepdim=True)
    dtorch_out, dtorch_indice = dtorch.max(dtorch_in, dim=1, keepdim=True)
    assert_tensor_equal(test_case, torch_out, dtorch_out)
    assert_tensor_equal(test_case, torch_indice, dtorch_indice)

    torch_out, torch_indice = torch_in.max(dim=1)
    dtorch_out, dtorch_indice = dtorch_in.max(dim=1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)
    assert_tensor_equal(test_case, torch_indice, dtorch_indice)

    torch_out, torch_indice = torch.min(torch_in, dim=1, keepdim=True)
    dtorch_out, dtorch_indice = dtorch.min(dtorch_in, dim=1, keepdim=True)
    assert_tensor_equal(test_case, torch_out, dtorch_out)
    assert_tensor_equal(test_case, torch_indice, dtorch_indice)


def _test_max_min_two_input(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float16, device=device)
    torch_other = torch.rand(*shape, dtype=torch.float16, device=device)
    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_other = dtorch.Tensor(torch_other)

    # Test max(input, other)
    torch_out = torch.max(torch_in, torch_other)
    dtorch_out = dtorch.max(dtorch_in, dtorch_other)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Test min(input, other)
    torch_out = torch.min(torch_in, torch_other)
    dtorch_out = dtorch.min(dtorch_in, dtorch_other)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Test tensor.max(other)
    torch_out = torch_in.max(torch_other)
    dtorch_out = dtorch_in.max(dtorch_other)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Test tensor.min(other)
    torch_out = torch_in.min(torch_other)
    dtorch_out = dtorch_in.min(dtorch_other)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Test max with broadcast
    torch_other_bc = torch.rand(shape[-1], dtype=torch.float16, device=device)
    dtorch_other_bc = dtorch.Tensor(torch_other_bc)
    torch_out = torch.max(torch_in, torch_other_bc)
    dtorch_out = dtorch.max(dtorch_in, dtorch_other_bc)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.min(torch_in, torch_other_bc)
    dtorch_out = dtorch.min(dtorch_in, dtorch_other_bc)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_arg_max_min(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float16, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    # Test dtorch.argmax function
    torch_out = torch.argmax(torch_in)
    dtorch_out = dtorch.argmax(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.argmax(torch_in, dim=1)
    dtorch_out = dtorch.argmax(dtorch_in, dim=1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.argmax(torch_in, dim=1, keepdim=True)
    dtorch_out = dtorch.argmax(dtorch_in, dim=1, keepdim=True)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Test Tensor.argmax method
    torch_out = torch_in.argmax()
    dtorch_out = dtorch_in.argmax()
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch_in.argmax(dim=1)
    dtorch_out = dtorch_in.argmax(dim=1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch_in.argmax(dim=1, keepdim=True)
    dtorch_out = dtorch_in.argmax(dim=1, keepdim=True)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Test dtorch.argmin function
    torch_out = torch.argmin(torch_in)
    dtorch_out = dtorch.argmin(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.argmin(torch_in, dim=1)
    dtorch_out = dtorch.argmin(dtorch_in, dim=1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.argmin(torch_in, dim=1, keepdim=True)
    dtorch_out = dtorch.argmin(dtorch_in, dim=1, keepdim=True)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Test Tensor.argmin method
    torch_out = torch_in.argmin()
    dtorch_out = dtorch_in.argmin()
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch_in.argmin(dim=1)
    dtorch_out = dtorch_in.argmin(dim=1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch_in.argmin(dim=1, keepdim=True)
    dtorch_out = dtorch_in.argmin(dim=1, keepdim=True)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestMaxMin(unittest.TestCase):
    def test_max_min(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_max_min(test_case, *arg)
            _test_arg_max_min(test_case, *arg)
            _test_max_min_two_input(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
