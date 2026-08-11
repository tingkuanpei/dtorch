"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_conv2d(
    test_case,
    device,
):
    torch_input = torch.randn(1, 4, 5, 5, dtype=torch.float32, device=device)
    torch_weight = torch.randn(8, 4, 3, 3, dtype=torch.float32, device=device)
    torch_bias = torch.randn(8, dtype=torch.float32, device=device)

    dtorch_input = dtorch.Tensor(torch_input)
    dtorch_weight = dtorch.Tensor(torch_weight)
    dtorch_bias = dtorch.Tensor(torch_bias)

    torch_output = torch.nn.functional.conv2d(torch_input, torch_weight, torch_bias, padding=1)
    dtorch_output = dtorch.nn.functional.conv2d(dtorch_input, dtorch_weight, dtorch_bias, padding=1)
    assert_tensor_equal(test_case, torch_output, dtorch_output)

    torch_output = torch.nn.functional.conv2d(torch_input, torch_weight, padding=2, dilation=4)
    dtorch_output = dtorch.nn.functional.conv2d(dtorch_input, dtorch_weight, padding=2, dilation=4)
    assert_tensor_equal(test_case, torch_output, dtorch_output)


def _test_distributed_conv2d(
    test_case,
    device,
):
    torch_input = torch.randn(1, 4, 16, 16, dtype=torch.float32, device=device)
    torch_weight = torch.randn(8, 4, 2, 2, dtype=torch.float32, device=device)
    torch_bias = torch.randn(8, dtype=torch.float32, device=device)

    device_mesh = dtorch.DeviceMesh(device, range(2))
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_input = dtorch.Tensor(torch_input, device_mesh=device_mesh, placements=[dtorch.Shard(3)])
    dtorch_weight = dtorch.Tensor(torch_weight, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_bias = dtorch.Tensor(torch_bias, device_mesh=device_mesh, placements=[dtorch.Replicate()])

    torch_output = torch.nn.functional.conv2d(torch_input, torch_weight, torch_bias, stride=2)
    dtorch_output = dtorch.nn.functional.conv2d(dtorch_input, dtorch_weight, dtorch_bias, stride=2)
    assert_tensor_allclose(test_case, torch_output, dtorch_output)

    m = dtorch.nn.Conv2d(
        in_channels=4,
        out_channels=8,
        kernel_size=2,
        stride=2,
        device_mesh=device_mesh,
    )
    m.weight.copy_(dtorch_weight)
    m.bias.copy_(dtorch_bias)
    dtorch_out = m(dtorch_input)
    assert_tensor_allclose(test_case, torch_output, dtorch_out)


class TestConv(unittest.TestCase):
    def test_conv2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            device = arg[0]
            _test_conv2d(test_case, device=device)
            _test_distributed_conv2d(test_case, device=device)


if __name__ == "__main__":
    unittest.main()
