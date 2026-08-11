"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_setitem_integer(test_case, shape, device):
    """Integer index with scalar value."""
    torch_t = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_t = dtorch.Tensor(torch_t.clone())

    torch_t[0] = 42.0
    dtorch_t[0] = 42.0
    assert_tensor_equal(test_case, torch_t, dtorch_t)


def _test_setitem_slice(test_case, shape, device):
    """Slice index with scalar and tensor values."""
    torch_t = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_t = dtorch.Tensor(torch_t.clone())

    # Scalar value
    torch_t[1:3] = 99.0
    dtorch_t[1:3] = 99.0
    assert_tensor_equal(test_case, torch_t, dtorch_t)

    # Tensor value (only for 1D tensors where slice dim matches)
    if len(shape) == 1:
        torch_t2 = torch.rand(*shape, dtype=torch.float32, device=device)
        dtorch_t2 = dtorch.Tensor(torch_t2.clone())
        value = torch.rand(3, dtype=torch.float32, device=device)
        torch_t2[0:3] = value
        dtorch_t2[0:3] = dtorch.Tensor(value)
        assert_tensor_equal(test_case, torch_t2, dtorch_t2)


def _test_setitem_ellipsis(test_case, shape, device):
    """Ellipsis index."""
    torch_t = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_t = dtorch.Tensor(torch_t.clone())

    torch_t[..., 0] = -1.0
    dtorch_t[..., 0] = -1.0
    assert_tensor_equal(test_case, torch_t, dtorch_t)


def _test_setitem_combined(test_case, shape, device):
    """Combined indices (integer + slice)."""
    if len(shape) < 3:
        return
    torch_t = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_t = dtorch.Tensor(torch_t.clone())

    torch_t[0, 1:3, :] = 77.0
    dtorch_t[0, 1:3, :] = 77.0
    assert_tensor_equal(test_case, torch_t, dtorch_t)


def _test_setitem_none_index(test_case, shape, device):
    """None (newaxis) index."""
    if len(shape) < 2:
        return
    torch_t = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_t = dtorch.Tensor(torch_t.clone())

    torch_t[None, :] = 5.0
    dtorch_t[None, :] = 5.0
    assert_tensor_equal(test_case, torch_t, dtorch_t)


def _test_setitem_tensor_value(test_case, shape, device):
    """Multi-dimensional tensor value."""
    if len(shape) < 3:
        return
    torch_t = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_t = dtorch.Tensor(torch_t.clone())

    # Assign a slice with a matching-shape tensor
    value_shape = shape[1:]  # e.g. for shape (4, 3, 2), value shape (3, 2)
    torch_v = torch.rand(*value_shape, dtype=torch.float32, device=device)
    dtorch_v = dtorch.Tensor(torch_v.clone())

    torch_t[0] = torch_v
    dtorch_t[0] = dtorch_v
    assert_tensor_equal(test_case, torch_t, dtorch_t)


def _test_setitem_negative_indices(test_case, shape, device):
    """Negative integer indices."""
    if len(shape) < 2:
        return
    torch_t = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_t = dtorch.Tensor(torch_t.clone())

    torch_t[-1, :] = 33.0
    dtorch_t[-1, :] = 33.0
    assert_tensor_equal(test_case, torch_t, dtorch_t)


def _test_setitem_chained(test_case, shape, device):
    """Chained operations: read then assign."""
    if len(shape) < 2:
        return
    torch_t = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_t = dtorch.Tensor(torch_t.clone())

    # t[0] = t[1] * 2
    torch_t[0] = torch_t[1] * 2
    dtorch_t[0] = dtorch_t[1] * 2
    assert_tensor_equal(test_case, torch_t, dtorch_t)


class TestSetItem(unittest.TestCase):
    def test_setitem_integer(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(8,), (4, 5), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_setitem_integer(test_case, *arg)

    def test_setitem_slice(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(8,), (4, 5), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_setitem_slice(test_case, *arg)

    def test_setitem_ellipsis(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 5), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_setitem_ellipsis(test_case, *arg)

    def test_setitem_combined(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 5), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_setitem_combined(test_case, *arg)

    def test_setitem_none(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 5), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_setitem_none_index(test_case, *arg)

    def test_setitem_tensor_value(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_setitem_tensor_value(test_case, *arg)

    def test_setitem_negative_indices(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 5), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_setitem_negative_indices(test_case, *arg)

    def test_setitem_chained(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 5), (2, 3, 4)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_setitem_chained(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
