"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_clone(test_case, shape, device, dtype):
    torch_in = torch.rand(*shape, device=device, dtype=dtype)
    dtorch_in = dtorch.Tensor(torch_in)
    device_mesh = dtorch.DeviceMesh(device, range(shape[0]))
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    placements = [dtorch.Shard(0)]
    dtorch_d_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=placements)

    # Test dtorch.clone() standalone function
    torch_out = torch.clone(torch_in)
    dtorch_out = dtorch.clone(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Verify clone is independent: modifying clone doesn't affect original
    dtorch_out_c = dtorch_out.to_torch()
    dtorch_out_c.add_(1.0)
    assert_tensor_equal(test_case, torch_in, dtorch_in)

    # Test Tensor.clone() method
    torch_out = torch_in.clone()
    dtorch_out = dtorch_in.clone()
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Verify method clone is independent
    dtorch_out_c = dtorch_out.to_torch()
    dtorch_out_c.add_(1.0)
    assert_tensor_equal(test_case, torch_in, dtorch_in)

    # Test distributed tensor clone
    torch_out = torch_in.clone()
    dtorch_out = dtorch.clone(dtorch_d_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    # Test distributed tensor method clone
    dtorch_out = dtorch_d_in.clone()
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


def _test_clone_int(test_case, shape, device):
    dtype = torch.int64
    torch_in = torch.randint(0, 100, shape, device=device, dtype=dtype)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch_in.clone()
    dtorch_out = dtorch_in.clone()
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Verify independence
    dtorch_out_c = dtorch_out.to_torch()
    dtorch_out_c.add_(1)
    assert_tensor_equal(test_case, torch_in, dtorch_in)


class TestClone(unittest.TestCase):
    def test_clone(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_clone(test_case, *arg)

    def test_clone_int(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_clone_int(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
