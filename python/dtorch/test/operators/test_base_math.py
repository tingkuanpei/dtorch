"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_base_math(test_case, shape, device, dtype):
    torch_in = 10 * torch.rand(*shape, device=device, dtype=dtype)
    dtorch_in = dtorch.Tensor(torch_in)
    device_mesh = dtorch.DeviceMesh(device, range(shape[0]))
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    placements = [dtorch.Shard(0)]
    dtorch_d_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=placements)

    def test_func_imp(func_name, in_tensor=None):
        if in_tensor is not None:
            t_in = in_tensor
            u_in = dtorch.Tensor(t_in)
            ud_in = dtorch.Tensor(t_in, device_mesh=device_mesh, placements=placements)
        else:
            t_in = torch_in
            u_in = dtorch_in
            ud_in = dtorch_d_in

        torch_func = getattr(torch, func_name)
        dtorch_func = getattr(dtorch, func_name)

        torch_out = torch_func(t_in)
        dtorch_out = dtorch_func(u_in)
        assert_tensor_equal(test_case, torch_out, dtorch_out)

        torch_out = getattr(t_in, func_name)()
        dtorch_out = getattr(u_in, func_name)()
        assert_tensor_equal(test_case, torch_out, dtorch_out)

        dtorch_out = dtorch_func(ud_in)
        assert_tensor_allclose(test_case, torch_out, dtorch_out)

    test_func_imp("exp")
    test_func_imp("square")
    test_func_imp("rsqrt")
    test_func_imp("abs")
    test_func_imp("round")
    test_func_imp("floor")
    test_func_imp("cos")
    test_func_imp("sin")
    test_func_imp("asin", 2 * torch.rand(*shape, device=device, dtype=dtype) - 1)
    test_func_imp("tanh")
    test_func_imp("neg")
    test_func_imp("reciprocal")
    test_func_imp("log")
    test_func_imp("log2")
    test_func_imp("log10")

    torch_out = -torch_in
    dtorch_out = -dtorch_in
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_in = torch.tensor([1, float("inf"), 2, float("-inf"), float("nan")])
    dtorch_in = dtorch.Tensor(torch_in)
    torch_out = torch.isinf(torch_in)
    dtorch_out = dtorch.isinf(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)
    torch_out = torch.isnan(torch_in)
    dtorch_out = dtorch.isnan(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestBaseMath(unittest.TestCase):
    def test_base_math(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_base_math(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
