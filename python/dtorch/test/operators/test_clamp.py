"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_clamp(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.clamp(torch_in, 0, 0.5)
    dtorch_out = dtorch.clamp(dtorch_in, 0, 0.5)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.clip(torch_in, 0, 0.5)
    dtorch_out = dtorch.clip(dtorch_in, 0, 0.5)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_clamp_tensor_min_max(test_case, shape, device, dtype):
    """Test clamp with tensor min/max of different dtypes (e.g. float16 input, float32 min/max)."""
    torch_in = torch.rand(*shape, dtype=dtype, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    # Create min/max tensors with float32 dtype (different from float16 input)
    min_val = torch.tensor(0.0, device=device)  # float32 by default
    max_val = torch.tensor(0.5, device=device)  # float32 by default
    dtorch_min = dtorch.Tensor(min_val)
    dtorch_max = dtorch.Tensor(max_val)

    torch_out = torch.clamp(torch_in, min_val, max_val)
    dtorch_out = dtorch.clamp(dtorch_in, dtorch_min, dtorch_max)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_clamp_tensor_min_max_broadcast(test_case, device, dtype):
    """Test clamp with broadcast-shaped min/max tensors of different dtypes."""
    shape = (2, 3, 4)
    torch_in = torch.rand(*shape, dtype=dtype, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    # broadcast-compatible min/max with different dtypes
    min_val = torch.tensor(0.0, device=device)  # scalar tensor, float32
    max_val = torch.tensor(0.5, device=device)  # scalar tensor, float32
    dtorch_min = dtorch.Tensor(min_val)
    dtorch_max = dtorch.Tensor(max_val)

    torch_out = torch.clamp(torch_in, min_val, max_val)
    dtorch_out = dtorch.clamp(dtorch_in, dtorch_min, dtorch_max)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestClamp(unittest.TestCase):
    def test_clamp(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 1, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_clamp(test_case, *arg)

    def test_clamp_tensor_min_max(test_case):
        """Test clamp with tensor min/max of different dtypes (e.g. float16 input, float32 min/max)."""
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3, 4), (3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float16, torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_clamp_tensor_min_max(test_case, *arg)

    def test_clamp_tensor_min_max_broadcast(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float16, torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_clamp_tensor_min_max_broadcast(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
