"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from collections import OrderedDict
import numbers
import unittest

import torch

import dtorch
from dtorch.test.test_util import (
    gen_arg_list,
    assert_tensor_allclose,
    assert_tensor_equal,
    check_pytorch_version,
)


def _test_layer_norm(test_case, shape, device, normalized_shape):
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    eps = 1e-5

    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    torch_scale = torch.rand(normalized_shape, dtype=torch.float32, device=device)
    torch_bias = torch.rand(normalized_shape, dtype=torch.float32, device=device)

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_scale = dtorch.Tensor(torch_scale)
    dtorch_bias = dtorch.Tensor(torch_bias)

    torch_out = torch.nn.functional.layer_norm(
        torch_in,
        normalized_shape,
        eps=eps,
    )
    dtorch_out = dtorch.nn.functional.layer_norm(
        dtorch_in,
        normalized_shape,
        eps=eps,
    )
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.layer_norm(
        torch_in,
        normalized_shape,
        torch_scale,
        torch_bias,
        eps=eps,
    )
    dtorch_out = dtorch.nn.functional.layer_norm(
        dtorch_in,
        normalized_shape,
        dtorch_scale,
        dtorch_bias,
        eps=eps,
    )
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    device_mesh = dtorch.DeviceMesh(device, range(shape[0]))
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
    dtorch_scale = dtorch.Tensor(torch_scale, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_bias = dtorch.Tensor(torch_bias, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_out = dtorch.nn.functional.layer_norm(
        dtorch_in,
        normalized_shape,
        dtorch_scale,
        dtorch_bias,
        eps=eps,
    )
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    m = dtorch.nn.LayerNorm(normalized_shape, eps, dtype=torch.float32, device_mesh=device_mesh)
    m.weight.copy_(dtorch_scale)
    m.bias.copy_(dtorch_bias)
    dtorch_out = m(dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


def _test_rms_norm(test_case, shape, device, normalized_shape):
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    eps = 1e-5

    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    torch_scale = torch.rand(normalized_shape, dtype=torch.float32, device=device)

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_scale = dtorch.Tensor(torch_scale)

    torch_out = torch.nn.functional.rms_norm(
        torch_in,
        normalized_shape,
        torch_scale,
        eps=eps,
    )
    dtorch_out = dtorch.nn.functional.rms_norm(
        dtorch_in,
        normalized_shape,
        dtorch_scale,
        eps=eps,
    )
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    device_mesh = dtorch.DeviceMesh(device, range(shape[0]))
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
    dtorch_scale = dtorch.Tensor(torch_scale, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_out = dtorch.nn.functional.rms_norm(
        dtorch_in,
        normalized_shape,
        dtorch_scale,
        eps=eps,
    )
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    m = dtorch.nn.RMSNorm(normalized_shape, eps, dtype=torch.float32, device_mesh=device_mesh)
    m.weight.copy_(dtorch_scale)
    dtorch_out = m(dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


def _test_group_norm(test_case, shape, device, num_groups):
    eps = 1e-5
    num_channels = shape[1]

    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    torch_scale = torch.rand(num_channels, dtype=torch.float32, device=device)
    torch_bias = torch.rand(num_channels, dtype=torch.float32, device=device)

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_scale = dtorch.Tensor(torch_scale)
    dtorch_bias = dtorch.Tensor(torch_bias)

    torch_out = torch.nn.functional.group_norm(
        torch_in,
        num_groups,
        eps=eps,
    )
    dtorch_out = dtorch.nn.functional.group_norm(
        dtorch_in,
        num_groups,
        eps=eps,
    )
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.group_norm(
        torch_in,
        num_groups,
        torch_scale,
        torch_bias,
        eps=eps,
    )
    dtorch_out = dtorch.nn.functional.group_norm(
        dtorch_in,
        num_groups,
        dtorch_scale,
        dtorch_bias,
        eps=eps,
    )
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    device_mesh = dtorch.DeviceMesh(device, range(shape[0]))
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
    dtorch_scale = dtorch.Tensor(torch_scale, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_bias = dtorch.Tensor(torch_bias, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_out = dtorch.nn.functional.group_norm(
        dtorch_in,
        num_groups,
        dtorch_scale,
        dtorch_bias,
        eps=eps,
    )
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    m = dtorch.nn.GroupNorm(num_groups, num_channels, eps, dtype=torch.float32, device_mesh=device_mesh)
    m.weight.copy_(dtorch_scale)
    m.bias.copy_(dtorch_bias)
    dtorch_out = m(dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


class TestNormalization(unittest.TestCase):
    def test_normalization(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 8)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["normalized_shape"] = [(8)]
        for arg in gen_arg_list(arg_dict):
            _test_layer_norm(test_case, *arg)
            if check_pytorch_version(min_version="2.5.0"):
                _test_rms_norm(test_case, *arg)

        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 8, 9)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["num_groups"] = [(4)]
        for arg in gen_arg_list(arg_dict):
            _test_group_norm(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
