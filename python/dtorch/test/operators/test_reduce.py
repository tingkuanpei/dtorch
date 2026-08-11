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


def _test_reduce(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float16, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.sum(torch_in)
    dtorch_out = dtorch.sum(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch_in.sum()
    dtorch_out = dtorch_in.sum()
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.sum(torch_in, dim=(0, 1), keepdim=True, dtype=torch.float32)
    dtorch_out = dtorch.sum(dtorch_in, dim=(0, 1), keepdim=True, dtype=torch.float32)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch_in.sum(dim=(0, 1), keepdim=True, dtype=torch.float32)
    dtorch_out = dtorch_in.sum(dim=(0, 1), keepdim=True, dtype=torch.float32)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.mean(torch_in, dim=(0, 1), keepdim=True, dtype=torch.float32)
    dtorch_out = dtorch.mean(dtorch_in, dim=(0, 1), keepdim=True, dtype=torch.float32)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_in = torch.arange(0, 3, device=device)
    dtorch_in = dtorch.arange(0, 3, device=device)

    torch_out = torch.any(torch_in)
    dtorch_out = dtorch.any(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.all(torch_in)
    dtorch_out = dtorch.all(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_distributed_reduce(test_case, shape, device):
    device_mesh = init_device_mesh(device, (2))
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[Shard(0)])

    torch_out = torch.sum(torch_in, dim=(1), dtype=torch.float32)
    dtorch_out = dtorch.sum(dtorch_in, dim=(1), dtype=torch.float32)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    torch_out = torch.sum(torch_in)
    dtorch_out = dtorch.sum(dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    torch_out = torch.sum(torch_in, dim=(0, 1), dtype=torch.float32)
    dtorch_out = dtorch.sum(dtorch_in, dim=(0, 1), dtype=torch.float32)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    torch_out = torch.sum(torch_in, dim=(2, 3), dtype=torch.float32)
    dtorch_out = dtorch.sum(dtorch_in, dim=(2, 3), dtype=torch.float32)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


class TestReduce(unittest.TestCase):
    def test_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 4, 8, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_reduce(test_case, *arg)
            _test_distributed_reduce(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
