"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal, assert_tensor_allclose


# ============================================================
# masked_fill tests
# ============================================================


def _test_masked_fill(test_case, shape, device, dtype):
    torch_in = torch.rand(*shape, dtype=dtype, device=device)
    mask = torch.rand(*shape, dtype=torch.float32, device=device) > 0.5
    fill_value = 3.14

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_mask = dtorch.Tensor(mask)

    torch_out = torch.masked_fill(torch_in, mask, fill_value)
    dtorch_out = dtorch_in.masked_fill(dtorch_mask, fill_value)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # Also test the functional form
    dtorch_out2 = dtorch.masked_fill(dtorch_in, dtorch_mask, fill_value)
    assert_tensor_equal(test_case, torch_out, dtorch_out2)


def _test_masked_fill_broadcast(test_case, shape, mask_shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    mask = torch.rand(*mask_shape, dtype=torch.float32, device=device) > 0.5
    fill_value = -1.0

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_mask = dtorch.Tensor(mask)

    torch_out = torch.masked_fill(torch_in, mask, fill_value)
    dtorch_out = dtorch_in.masked_fill(dtorch_mask, fill_value)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_masked_fill_all_false(test_case, shape, device):
    """mask all False: output should equal input."""
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    mask = torch.zeros(*shape, dtype=torch.bool, device=device)
    fill_value = 42.0

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_mask = dtorch.Tensor(mask)

    torch_out = torch.masked_fill(torch_in, mask, fill_value)
    dtorch_out = dtorch_in.masked_fill(dtorch_mask, fill_value)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_masked_fill_all_true(test_case, shape, device):
    """mask all True: all elements should be fill_value."""
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    mask = torch.ones(*shape, dtype=torch.bool, device=device)
    fill_value = -999.0

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_mask = dtorch.Tensor(mask)

    torch_out = torch.masked_fill(torch_in, mask, fill_value)
    dtorch_out = dtorch_in.masked_fill(dtorch_mask, fill_value)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_masked_fill_inplace(test_case, shape, device):
    """Test masked_fill_ (in-place variant)."""
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    mask = torch.rand(*shape, dtype=torch.float32, device=device) > 0.5
    fill_value = 7.7

    dtorch_in = dtorch.Tensor(torch_in.clone())
    dtorch_mask = dtorch.Tensor(mask)

    torch_in.masked_fill_(mask, fill_value)
    dtorch_in.masked_fill_(dtorch_mask, fill_value)
    # After in-place, dtorch tensor should match the (now-mutated) torch tensor
    assert_tensor_equal(test_case, torch_in, dtorch_in)


def _test_masked_fill_distributed(test_case, shape, mesh_size, device):
    device_mesh = dtorch.DeviceMesh(device, range(mesh_size))
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    mask = torch.rand(*shape, dtype=torch.float32, device=device) > 0.5
    fill_value = 0.0

    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
    dtorch_mask = dtorch.Tensor(mask, device_mesh=device_mesh, placements=[dtorch.Shard(0)])

    torch_out = torch.masked_fill(torch_in, mask, fill_value)
    dtorch_out = dtorch_in.masked_fill(dtorch_mask, fill_value)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


# ============================================================
# masked_scatter tests
# ============================================================


def _test_masked_scatter(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    mask = torch.rand(*shape, dtype=torch.float32, device=device) > 0.5
    # source must have enough elements to fill all True positions
    num_true = mask.sum().item()
    source = torch.rand(num_true + 5, dtype=torch.float32, device=device)  # extra elements

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_mask = dtorch.Tensor(mask)
    dtorch_source = dtorch.Tensor(source)

    torch_out = torch.masked_scatter(torch_in, mask, source)
    dtorch_out = dtorch_in.masked_scatter(dtorch_mask, dtorch_source)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_masked_scatter_exact(test_case, shape, device):
    """source has exactly the right number of elements."""
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    mask = torch.rand(*shape, dtype=torch.float32, device=device) > 0.5
    num_true = mask.sum().item()
    source = torch.rand(num_true, dtype=torch.float32, device=device)

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_mask = dtorch.Tensor(mask)
    dtorch_source = dtorch.Tensor(source)

    torch_out = torch.masked_scatter(torch_in, mask, source)
    dtorch_out = dtorch_in.masked_scatter(dtorch_mask, dtorch_source)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


# ============================================================
# Test classes
# ============================================================


class TestMaskedFill(unittest.TestCase):
    def test_masked_fill(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(3,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32, torch.float16]
        for arg in gen_arg_list(arg_dict):
            _test_masked_fill(test_case, *arg)

    def test_masked_fill_broadcast(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape, mask_shape"] = [((2, 3, 4), (3, 4)), ((2, 3, 4), (4,)), ((2, 3, 4), (1, 4))]
        arg_dict["device"] = ["cpu", "cuda"]
        for args in gen_arg_list(arg_dict):
            shape, mask_shape = args[0]
            device = args[1]
            _test_masked_fill_broadcast(test_case, shape, mask_shape, device)

    def test_masked_fill_all_false(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_masked_fill_all_false(test_case, *arg)

    def test_masked_fill_all_true(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_masked_fill_all_true(test_case, *arg)

    def test_masked_fill_inplace(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_masked_fill_inplace(test_case, *arg)

    def test_masked_fill_distributed(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape, mesh_size"] = [((4, 8), 2), ((8, 16), 2)]
        arg_dict["device"] = ["cpu", "cuda"]
        for args in gen_arg_list(arg_dict):
            shape, mesh_size = args[0]
            device = args[1]
            _test_masked_fill_distributed(test_case, shape, mesh_size, device)


class TestMaskedScatter(unittest.TestCase):
    def test_masked_scatter(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_masked_scatter(test_case, *arg)

    def test_masked_scatter_exact(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_masked_scatter_exact(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
