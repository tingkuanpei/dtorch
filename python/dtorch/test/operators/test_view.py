"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.distributed_spec import init_device_mesh, Replicate, Partial
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_view(test_case, shape, device):
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch_in.view(3, -1)
    dtorch_out = dtorch_in.view(3, -1)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_distributed_view(test_case, shape, device):
    device_mesh = init_device_mesh(device, (2))
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[Replicate()])
    dtorch_in = dtorch_in.view([Partial()])
    assert_tensor_allclose(test_case, 2 * torch_in, dtorch_in)

    dtorch_in = dtorch_in.view([Replicate()])
    assert_tensor_allclose(test_case, torch_in, dtorch_in)


class TestView(unittest.TestCase):
    def test_view(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_view(test_case, *arg)
            _test_distributed_view(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
