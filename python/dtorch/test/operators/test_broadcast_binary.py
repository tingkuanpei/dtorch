"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal
from dtorch.distributed_spec import init_device_mesh, Replicate, Shard


def _test_broadcast_binary(test_case, device):
    def testFunc(
        test_cast,
        x_shape,
        y_shape,
        device,
        mesh_size=None,
        placement_a=None,
        placement_b=None,
    ):
        torch_x = torch.rand(*x_shape, dtype=torch.float32, device=device)
        torch_y = torch.rand(*y_shape, dtype=torch.float32, device=device)
        dtorch_x = dtorch.Tensor(torch_x)
        dtorch_y = dtorch.Tensor(torch_y)

        torch_z = torch_x + torch_y
        dtorch_z = dtorch_x + dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x + 2
        dtorch_z = dtorch_x + 2
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = 1 + torch_y
        dtorch_z = 1 + dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x - torch_y
        dtorch_z = dtorch_x - dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = 1 - torch_y
        dtorch_z = 1 - dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x - 1
        dtorch_z = dtorch_x - 1
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x * torch_y
        dtorch_z = dtorch_x * dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x * 2
        dtorch_z = dtorch_x * 2
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = 4.2 * torch_y
        dtorch_z = 4.2 * dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x / torch_y
        dtorch_z = dtorch_x / dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x / 2.3
        dtorch_z = dtorch_x / 2.3
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = 4.1 / torch_y
        dtorch_z = 4.1 / dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch.pow(4.1, torch_y)
        dtorch_z = dtorch.pow(4.1, dtorch_y)
        assert_tensor_allclose(test_case, torch_z, dtorch_z)
        dtorch_z = 4.1**dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch.pow(torch_x, 4.1)
        dtorch_z = dtorch.pow(dtorch_x, 4.1)
        assert_tensor_allclose(test_case, torch_z, dtorch_z)
        dtorch_z = dtorch_x**4.1
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch.pow(torch_x, torch_y)
        dtorch_z = dtorch.pow(dtorch_x, dtorch_y)
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x == torch_y
        dtorch_z = dtorch_x == dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x == 4.1
        dtorch_z = dtorch_x == 4.1
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = 4.1 == torch_x
        dtorch_z = 4.1 == dtorch_x
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x > torch_y
        dtorch_z = dtorch_x > dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x > 4.1
        dtorch_z = dtorch_x > 4.1
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = 4.1 > torch_x
        dtorch_z = 4.1 > dtorch_x
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x >= torch_y
        dtorch_z = dtorch_x >= dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x >= 4.1
        dtorch_z = dtorch_x >= 4.1
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x < torch_y
        dtorch_z = dtorch_x < dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x < 4.1
        dtorch_z = dtorch_x < 4.1
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x <= torch_y
        dtorch_z = dtorch_x <= dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        torch_z = torch_x <= 4.1
        dtorch_z = dtorch_x <= 4.1
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        if mesh_size is not None:
            placement_b = placement_b if placement_b is not None else placement_a
            device_mesh = dtorch.DeviceMesh(device, range(mesh_size))
            if not dtorch.default_graph.satisfy(device_mesh):
                return
            dtorch_x = dtorch.Tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[placement_a],
            )
            dtorch_y = dtorch.Tensor(
                torch_y,
                device_mesh=device_mesh,
                placements=[placement_b],
            )

            torch_z = torch_x + torch_y
            dtorch_z = dtorch_x + dtorch_y
            assert_tensor_allclose(test_case, torch_z, dtorch_z)

            torch_z = torch_x - torch_y
            dtorch_z = dtorch_x - dtorch_y
            assert_tensor_allclose(test_case, torch_z, dtorch_z)

            torch_z = torch_x * torch_y
            dtorch_z = dtorch_x * dtorch_y
            assert_tensor_allclose(test_case, torch_z, dtorch_z)

            torch_z = torch_x / torch_y
            dtorch_z = dtorch_x / dtorch_y
            assert_tensor_allclose(test_case, torch_z, dtorch_z)

    x_shape = [2, 3, 5]
    y_shape = [2, 3, 5]
    testFunc(test_case, x_shape, y_shape, device, 2, Shard(0))

    x_shape = [3, 3, 5]
    y_shape = [3, 1]
    testFunc(test_case, x_shape, y_shape, device, 3, Shard(1), Shard(0))

    x_shape = [3, 1]
    y_shape = [2, 3, 5]
    testFunc(test_case, x_shape, y_shape, device, 2, Replicate(), Shard(0))

    x_shape = [4, 1]
    y_shape = [2, 3, 1, 5]
    testFunc(test_case, x_shape, y_shape, device, 4, Replicate(), Replicate())

    x_shape = [2, 3, 4, 1]
    y_shape = [1, 3, 1, 5]
    testFunc(test_case, x_shape, y_shape, device, 3, Shard(1))

    torch_x = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
    torch_y = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
    dtorch_x = dtorch.Tensor(torch_x)
    dtorch_y = dtorch.Tensor(torch_y)

    torch_z = torch.logical_and(torch_x, torch_y)
    dtorch_z = dtorch.logical_and(dtorch_x, dtorch_y)
    assert_tensor_allclose(test_case, torch_z, dtorch_z)

    torch_z = torch.logical_or(torch_x, torch_y)
    dtorch_z = dtorch.logical_or(dtorch_x, dtorch_y)
    assert_tensor_allclose(test_case, torch_z, dtorch_z)

    torch_x = torch.tensor([0, 1, 10, 0], dtype=torch.long)
    torch_z = torch_x * 10
    dtorch_x = dtorch.Tensor(torch_x)
    dtorch_z = dtorch_x * 10
    assert_tensor_allclose(test_case, torch_z, dtorch_z)


def _test_broadcast_binary_min_max(test_case, device):
    def testFunc(test_cast, x_shape, y_shape, device, mesh_size=None, placement_a=None, placement_b=None):
        torch_x = torch.rand(*x_shape, dtype=torch.float32, device=device)
        torch_y = torch.rand(*y_shape, dtype=torch.float32, device=device)
        dtorch_x = dtorch.Tensor(torch_x)
        dtorch_y = dtorch.Tensor(torch_y)

        # Test minimum
        torch_z = torch.minimum(torch_x, torch_y)
        dtorch_z = dtorch.minimum(dtorch_x, dtorch_y)
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        # Test maximum
        torch_z = torch.maximum(torch_x, torch_y)
        dtorch_z = dtorch.maximum(dtorch_x, dtorch_y)
        assert_tensor_allclose(test_case, torch_z, dtorch_z)

        if mesh_size is not None:
            placement_b = placement_b if placement_b is not None else placement_a
            device_mesh = dtorch.DeviceMesh(device, range(mesh_size))
            if not dtorch.default_graph.satisfy(device_mesh):
                return
            dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=[placement_a])
            dtorch_y = dtorch.Tensor(torch_y, device_mesh=device_mesh, placements=[placement_b])

            torch_z = torch.minimum(torch_x, torch_y)
            dtorch_z = dtorch.minimum(dtorch_x, dtorch_y)
            assert_tensor_allclose(test_case, torch_z, dtorch_z)

            torch_z = torch.maximum(torch_x, torch_y)
            dtorch_z = dtorch.maximum(dtorch_x, dtorch_y)
            assert_tensor_allclose(test_case, torch_z, dtorch_z)

    # Same shape
    x_shape = [2, 3, 5]
    y_shape = [2, 3, 5]
    testFunc(test_case, x_shape, y_shape, device, 2, Shard(0))

    # Broadcast
    x_shape = [3, 3, 5]
    y_shape = [3, 1]
    testFunc(test_case, x_shape, y_shape, device, 3, Shard(1), Shard(0))

    x_shape = [3, 1]
    y_shape = [2, 3, 5]
    testFunc(test_case, x_shape, y_shape, device, 2, Replicate(), Shard(0))

    # Scalar broadcast
    x_shape = [4, 1]
    y_shape = [2, 3, 1, 5]
    testFunc(test_case, x_shape, y_shape, device, 4, Replicate(), Replicate())

    x_shape = [2, 3, 4, 1]
    y_shape = [1, 3, 1, 5]
    testFunc(test_case, x_shape, y_shape, device, 3, Shard(1))


def _test_broadcast_binary_with_replicate_shard(test_case, device):
    x_shape = [3, 4, 5]
    y_shape = [4, 1]
    device_mesh = init_device_mesh(device, (2, 2))
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    torch_x = torch.rand(*x_shape, dtype=torch.float16, device=device)
    torch_y = torch.rand(*y_shape, dtype=torch.float16, device=device)
    dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=[Shard(1), Shard(1)])
    dtorch_y = dtorch.Tensor(torch_y, device_mesh=device_mesh, placements=[Replicate(), Replicate()])

    torch_z = torch_x + torch_y
    dtorch_z = dtorch_x + dtorch_y
    assert_tensor_allclose(test_case, torch_z, dtorch_z)


def _test_broadcast_binary_with_replicate_sub_shard(test_case, device):
    device_mesh = init_device_mesh(device, (1, 1, 2))
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    torch_in_a = torch.rand((1, 24, 512, 128), dtype=torch.bfloat16, device=device)
    torch_in_b = torch.rand((1, 24, 4096, 128), dtype=torch.bfloat16, device=device)
    torch_in_c = torch.rand((1, 1, 4608, 128), dtype=torch.float32, device=device)
    torch_out = torch.cat((torch_in_a, torch_in_b), dim=2)
    torch_out = torch_out.float() * torch_in_c

    dtorch_in_a = dtorch.Tensor(torch_in_a, device_mesh=device_mesh, placements=[Replicate(), Replicate(), Shard(2)])
    dtorch_in_b = dtorch.Tensor(torch_in_b, device_mesh=device_mesh, placements=[Replicate(), Replicate(), Shard(2)])
    dtorch_in_c = dtorch.Tensor(torch_in_c, device_mesh=device_mesh, placements=[Replicate(), Replicate(), Replicate()])
    dtorch_out = dtorch.cat((dtorch_in_a, dtorch_in_b), dim=2)
    dtorch_out = dtorch_out.float() * dtorch_in_c
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


def _test_broadcast_binary_data_type_promotion(test_case, device):
    x_shape = [3, 3, 5]
    y_shape = [3, 1]

    torch_x = torch.rand(*x_shape, dtype=torch.float16, device=device)
    torch_y = torch.rand(*y_shape, dtype=torch.float32, device=device)
    dtorch_x = dtorch.Tensor(torch_x)
    dtorch_y = dtorch.Tensor(torch_y)

    torch_z = torch_x + torch_y
    dtorch_z = dtorch_x + dtorch_y
    assert_tensor_allclose(test_case, torch_z, dtorch_z)

    torch_z = torch_x * torch_y
    dtorch_z = dtorch_x * dtorch_y
    assert_tensor_allclose(test_case, torch_z, dtorch_z)

    torch_x = torch.rand(*x_shape, dtype=torch.float32, device=device)
    torch_y = torch.rand(*y_shape, dtype=torch.float16, device=device)
    dtorch_x = dtorch.Tensor(torch_x)
    dtorch_y = dtorch.Tensor(torch_y)

    torch_z = torch_x * torch_y
    dtorch_z = dtorch_x * dtorch_y
    assert_tensor_allclose(test_case, torch_z, dtorch_z)


def _test_equal(test_case, device):
    def testFunc(
        test_cast, x_shape, y_shape, device, equal_expected, mesh_size=None, placement_a=None, placement_b=None
    ):
        torch_x = torch.rand(*x_shape, dtype=torch.float32, device=device)
        if equal_expected:
            torch_y = torch_x.clone()
        else:
            torch_y = torch.rand(*y_shape, dtype=torch.float32, device=device)

        dtorch_x = dtorch.Tensor(torch_x)
        dtorch_y = dtorch.Tensor(torch_y)

        torch_result = torch_x.equal(torch_y)
        dtorch_result = dtorch_x.equal(dtorch_y)
        test_cast.assertEqual(torch_result, dtorch_result)

        if mesh_size is not None:
            placement_b = placement_b if placement_b is not None else placement_a
            device_mesh = dtorch.DeviceMesh(device, range(mesh_size))
            if not dtorch.default_graph.satisfy(device_mesh):
                return
            dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=[placement_a])
            dtorch_y = dtorch.Tensor(torch_y, device_mesh=device_mesh, placements=[placement_b])

            torch_result = torch_x.equal(torch_y)
            dtorch_result = dtorch_x.equal(dtorch_y)
            test_cast.assertEqual(torch_result, dtorch_result)

    # Same tensors → True
    x_shape = [2, 3, 5]
    y_shape = [2, 3, 5]
    testFunc(test_case, x_shape, y_shape, device, True, 2, Shard(0))

    # Different values → False
    x_shape = [3, 3, 5]
    y_shape = [3, 3, 5]
    testFunc(test_case, x_shape, y_shape, device, False, 3, Shard(1))

    # Different shapes → False
    x_shape = [3, 4, 5]
    y_shape = [3, 4, 6]
    testFunc(test_case, x_shape, y_shape, device, False, 2, Replicate(), Replicate())

    # Same tensors, broadcast shapes → different shapes → False
    x_shape = [3, 1]
    y_shape = [2, 3, 5]
    testFunc(test_case, x_shape, y_shape, device, False, 2, Replicate(), Shard(0))

    # Same tensors, different placements (Replicate + Shard)
    x_shape = [2, 3, 4, 1]
    y_shape = [2, 3, 4, 1]
    testFunc(test_case, x_shape, y_shape, device, True, 3, Shard(1), Replicate())


class TestAdd(unittest.TestCase):
    def test_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_broadcast_binary(test_case, *arg)
            _test_broadcast_binary_data_type_promotion(test_case, *arg)
            _test_broadcast_binary_with_replicate_shard(test_case, *arg)
            _test_broadcast_binary_with_replicate_sub_shard(test_case, *arg)
            _test_broadcast_binary_min_max(test_case, *arg)
            _test_equal(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
