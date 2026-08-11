"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.distributed_spec import init_device_mesh
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_linear(test_case, m, n, k, device):
    torch_in = torch.rand(m, k, dtype=torch.float32, device=device)
    torch_weight = torch.rand(n, k, dtype=torch.float32, device=device)
    torch_bias = torch.rand(n, dtype=torch.float32, device=device)

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_weight = dtorch.Tensor(torch_weight)
    dtorch_bias = dtorch.Tensor(torch_bias)

    torch_out = torch.nn.functional.linear(torch_in, torch_weight, torch_bias)
    dtorch_out = dtorch.nn.functional.linear(dtorch_in, dtorch_weight, dtorch_bias)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.linear(torch_in, torch_weight)
    dtorch_out = dtorch.nn.functional.linear(dtorch_in, dtorch_weight)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_distributed_linear(test_case, m, n, k, device):
    torch_in = torch.rand(m, k, dtype=torch.float32, device=device)
    torch_weight = torch.rand(n, k, dtype=torch.float32, device=device)
    torch_bias = torch.rand(n, dtype=torch.float32, device=device)

    device_mesh = dtorch.DeviceMesh(
        device,
        range(m),
        mesh_dim_names=[
            "tp",
        ],
    )
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
    dtorch_weight = dtorch.Tensor(torch_weight, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_bias = dtorch.Tensor(torch_bias, device_mesh=device_mesh, placements=[dtorch.Replicate()])

    torch_out = torch.nn.functional.linear(torch_in, torch_weight, torch_bias)
    dtorch_out = dtorch.nn.functional.linear(dtorch_in, dtorch_weight, dtorch_bias)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.linear(torch_in, torch_weight)
    dtorch_out = dtorch.nn.functional.linear(dtorch_in, dtorch_weight)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    device_mesh = dtorch.DeviceMesh(
        device,
        range(n),
        mesh_dim_names=[
            "tp",
        ],
    )
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_weight = dtorch.Tensor(torch_weight, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
    dtorch_bias = dtorch.Tensor(torch_bias, device_mesh=device_mesh, placements=[dtorch.Shard(0)])

    torch_out = torch.nn.functional.linear(torch_in, torch_weight, torch_bias)
    dtorch_out = dtorch.nn.functional.linear(dtorch_in, dtorch_weight, dtorch_bias)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    m = dtorch.nn.ColumnParallelLinear(
        in_features=k,
        out_features=n,
        bias=True,
        dtype=torch.float32,
        device_mesh=device_mesh,
    )
    m.weight.copy_(dtorch_weight)
    m.bias.copy_(dtorch_bias)
    dtorch_out = m(dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    device_mesh = dtorch.DeviceMesh(
        device,
        range(1),
        mesh_dim_names=[
            "tp",
        ],
    )
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_weight = dtorch.Tensor(torch_weight, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
    dtorch_bias = dtorch.Tensor(torch_bias, device_mesh=device_mesh, placements=[dtorch.Shard(0)])

    m = dtorch.nn.RowParallelLinear(
        in_features=k,
        out_features=n,
        bias=True,
        dtype=torch.float32,
        device_mesh=device_mesh,
    )
    m.weight.copy_(dtorch_weight)
    m.bias.copy_(dtorch_bias)
    dtorch_out = m(dtorch_in)
    torch_out = torch.nn.functional.linear(torch_in, torch_weight, torch_bias)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


def _test_distributed_linear_with_cp(test_case, m, n, k, device):
    torch_in = torch.rand(m, 16, k, dtype=torch.float32, device=device)
    torch_weight = torch.rand(n, k, dtype=torch.float32, device=device)
    torch_bias = torch.rand(n, dtype=torch.float32, device=device)
    torch_out = torch.nn.functional.linear(torch_in, torch_weight, torch_bias)

    ulysess_cp = 2
    ring_cp = 4
    tp = n

    device_mesh = init_device_mesh(
        device,
        (ulysess_cp, ring_cp, tp),
        mesh_dim_names=["ulysess_cp", "ring_cp", "tp"],
    )
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_in = dtorch.Tensor(
        torch_in,
        device_mesh=device_mesh,
        placements=[dtorch.Shard(1), dtorch.Shard(1), dtorch.Replicate()],
    )

    m = dtorch.nn.ColumnParallelLinear(
        in_features=k,
        out_features=n,
        bias=True,
        dtype=torch.float32,
        device_mesh=device_mesh,
    )
    m.weight.copy_(dtorch.Tensor(torch_weight))
    m.bias.copy_(dtorch.Tensor(torch_bias))
    dtorch_out = m(dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


class TestLinear(unittest.TestCase):
    def test_linear(test_case):
        arg_dict = OrderedDict()
        arg_dict["m"] = [2]
        arg_dict["n"] = [4]
        arg_dict["k"] = [8]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_linear(test_case, *arg)
            _test_distributed_linear(test_case, *arg)
            _test_distributed_linear_with_cp(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
