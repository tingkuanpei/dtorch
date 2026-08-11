"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal
from dtorch.distributed_spec import init_device_mesh


def _test_embedding(test_case, shape, device, dtype):
    num_embeddings = 600
    embedding_dim = 100

    torch_in = torch.randint(
        low=0,
        high=num_embeddings,
        size=shape,
        device=device,
    )
    torch_weight = torch.rand(num_embeddings, embedding_dim, device=device, dtype=dtype)
    torch_out = torch.nn.functional.embedding(torch_in, torch_weight)

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_weight = dtorch.Tensor(torch_weight)
    dtorch_out = dtorch.nn.functional.embedding(dtorch_in, dtorch_weight)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    device_mesh = dtorch.DeviceMesh(
        device,
        range(4),
        mesh_dim_names=[
            "tp",
        ],
    )
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_in = dtorch.Tensor(torch_in, device_mesh=device_mesh, placements=[dtorch.Replicate()])
    dtorch_weight = dtorch.Tensor(torch_weight, device_mesh=device_mesh, placements=[dtorch.Shard(1)])
    dtorch_out = dtorch.nn.functional.embedding(dtorch_in, dtorch_weight)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    m = dtorch.nn.Embedding(num_embeddings, embedding_dim, device_mesh=device_mesh)
    m.weight.copy_(dtorch_weight)
    dtorch_out = m(dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


def _test_embedding_with_cp(test_case, shape, device, dtype):
    ulysess_cp = 2
    ring_cp = 4
    tp = 2
    num_embeddings = 600
    embedding_dim = 100

    torch_in = torch.randint(
        low=0,
        high=num_embeddings,
        size=shape,
        device=device,
    )
    torch_weight = torch.rand(num_embeddings, embedding_dim, device=device, dtype=dtype)
    torch_out = torch.nn.functional.embedding(torch_in, torch_weight)

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

    m = dtorch.nn.Embedding(
        num_embeddings,
        embedding_dim,
        device_mesh=device_mesh,
    )
    m.weight.copy_(dtorch.Tensor(torch_weight))
    dtorch_out = m(dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


class TestEmbedding(unittest.TestCase):
    def test_embedding(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_embedding(test_case, *arg)

        arg_dict["shape"] = [(2, 64)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_embedding_with_cp(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
