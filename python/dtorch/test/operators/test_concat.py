"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.distributed_spec import init_device_mesh, Replicate, Shard
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_concat(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.concat((torch_in, torch_in, torch_in), dim=1)
    dtorch_out = dtorch.concat((dtorch_in, dtorch_in, dtorch_in), dim=1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.stack((torch_in, torch_in, torch_in), dim=0)
    dtorch_out = dtorch.stack((dtorch_in, dtorch_in, dtorch_in), dim=0)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.stack((torch_in, torch_in, torch_in), dim=4)
    dtorch_out = dtorch.stack((dtorch_in, dtorch_in, dtorch_in), dim=4)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.stack((torch_in, torch_in), dim=-1)
    dtorch_out = dtorch.stack((dtorch_in, dtorch_in), dim=-1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.stack((torch_in, torch_in), dim=-5)
    dtorch_out = dtorch.stack((dtorch_in, dtorch_in), dim=-5)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_distributed_concat(test_case, shape, device):
    # 1D
    device_mesh = init_device_mesh(device, (2))
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[Shard(1)])

    torch_out = torch.concat((torch_in, torch_in), dim=1)
    dtorch_out = dtorch.concat((dtorch_in, dtorch_in), dim=1)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    shape[1] = shape[1] * 2
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=dtorch_out.device_mesh, placements=dtorch_out.placements)
    assert_tensor_allclose(test_case, torch_in, dtorch_in)

    dtorch_out = dtorch_in + dtorch_in
    assert_tensor_allclose(test_case, 2 * torch_in, dtorch_out)

    # 2D
    device_mesh = init_device_mesh(device, (2, 2))
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[Shard(1), Shard(1)])

    torch_out = torch.concat((torch_in, torch_in), dim=1)
    dtorch_out = dtorch.concat((dtorch_in, dtorch_in), dim=1)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    dtorch_out = dtorch.Tensor(torch_out, device_mesh=device_mesh, placements=dtorch_out.placements)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    dtorch_out = dtorch_out.redistribute_by_dict(
        placements_dict={
            0: Replicate(),
            1: Replicate(),
        },
        default_placement_mode="keep",
    )

    assert_tensor_allclose(test_case, torch_out, dtorch_out)


class TestConcat(unittest.TestCase):
    def test_concat(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [[3, 8, 4, 5]]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_concat(test_case, *arg)
            _test_distributed_concat(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
