"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest

import torch

import dtorch
from dtorch import Replicate, Shard, Partial, Graph
from dtorch.distributed_spec import init_device_mesh
from dtorch.test.test_util import assert_tensor_allclose


class TestDistributedTensor(unittest.TestCase):
    def test_distributed_tensor_create_api(test_case):
        x_shape = [8, 7, 4, 6]

        # from_torch
        device_mesh = dtorch.DeviceMesh("cpu", [0, 1, 2, 3])
        placements = [Shard(0)]
        torch_x = torch.rand(x_shape, dtype=torch.float32, device="cpu")
        dtorch_x = dtorch.from_torch(torch_x, device_mesh=device_mesh, placements=placements)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        # zeros
        torch_x = torch.zeros(x_shape, dtype=torch.float32, device="cpu")
        dtorch_x = dtorch.zeros(x_shape, dtype=torch.float32, device_mesh=device_mesh, placements=placements)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        dtorch_x = dtorch.zeros(x_shape, dtype=torch.float32, device_mesh=device_mesh)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        # randn
        torch_generator = torch.Generator().manual_seed(42)
        dtorch_generator = torch.Generator().manual_seed(42)
        torch_x = torch.randn(*x_shape, generator=torch_generator)
        dtorch_x = dtorch.randn(
            *x_shape,
            generator=dtorch_generator,
            device_mesh=device_mesh,
            placements=placements,
        )
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        # empty
        device_mesh = init_device_mesh("cpu", (2, 1, 1))
        placements = [Replicate(), Shard(1), Replicate()]
        torch_generator = torch.Generator().manual_seed(42)
        dtorch_generator = torch.Generator().manual_seed(42)
        dtorch_x = dtorch.empty(
            *x_shape,
            device_mesh=device_mesh,
            placements=placements,
        )

        test_case.assertTrue(dtorch_x.placements[0] == Replicate())
        test_case.assertTrue(dtorch_x.placements[1] == Replicate())
        test_case.assertTrue(dtorch_x.placements[2] == Replicate())

    def test_distributed_tensor_constructor(test_case):
        x_shape = [4, 8]
        device_mesh = dtorch.DeviceMesh("cpu", [0, 1, 2, 3])
        torch_x = torch.rand(*x_shape, dtype=torch.float32)

        # S0
        placements = [Shard(0)]
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=placements)

        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        test_case.assertTrue(dtorch_x.dtype == torch_x.dtype)
        test_case.assertTrue(dtorch_x.device_mesh == device_mesh)
        test_case.assertTrue(dtorch_x.placements == placements)

        # S1
        placements = [Shard(1)]
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=placements)

        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        test_case.assertTrue(dtorch_x.dtype == torch_x.dtype)
        test_case.assertTrue(dtorch_x.device_mesh == device_mesh)
        test_case.assertTrue(dtorch_x.placements == placements)

        # R
        placements = [Replicate()]
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=placements)

        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        test_case.assertTrue(dtorch_x.dtype == torch_x.dtype)
        test_case.assertTrue(dtorch_x.device_mesh == device_mesh)
        test_case.assertTrue(dtorch_x.placements == placements)

        # P
        placements = [dtorch.Partial()]
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=placements)

        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        test_case.assertTrue(dtorch_x.dtype == torch_x.dtype)
        test_case.assertTrue(dtorch_x.device_mesh == device_mesh)
        test_case.assertTrue(dtorch_x.placements == placements)

        # Not equal spilt
        x_shape = [4, 11]
        device_mesh = dtorch.DeviceMesh("cpu", [0, 1, 2, 3])
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        placements = [Shard(1)]
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=placements)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        test_case.assertTrue(dtorch_x.dtype == torch_x.dtype)
        test_case.assertTrue(dtorch_x.device_mesh == device_mesh)
        test_case.assertTrue(dtorch_x.placements == placements)

        x_shape = [6, 123]
        device_mesh = init_device_mesh("cpu", (3, 2, 4, 3, 6))
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        placements = [Replicate(), Shard(1), Partial(), Shard(1), Shard(0)]
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=placements)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        x_shape = [2, 4096, 24, 64]
        device_mesh = init_device_mesh("cpu", (2, 2, 2))
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        placements = [Shard(0), Shard(1), Shard(1)]
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=placements)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

    def test_distributed_tensor_to(test_case):
        x_shape = [4, 8]
        cpu_device_mesh = dtorch.DeviceMesh("cpu", [0, 1])
        other_cpu_device_mesh = dtorch.DeviceMesh("cpu", [2, 3])

        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=cpu_device_mesh)

        dtorch_x = dtorch_x.to(other_cpu_device_mesh)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        gpu_device_mesh = dtorch.DeviceMesh("cuda", [0, 1])
        if not dtorch.default_graph.satisfy(gpu_device_mesh):
            return
        dtorch_x = dtorch_x.to(gpu_device_mesh)
        torch_x = torch_x.to("cuda")
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

    def test_distributed_tensor_redistribute(test_case):
        def test_distributed_uneven_split_tensor_redistribute_imp(test_case, device, graph):
            # 1D
            x_shape = [5, 9]
            torch_x = torch.rand(*x_shape, device=device, dtype=torch.float32)
            x_local = dtorch.tensor(torch_x, graph=graph)
            assert_tensor_allclose(test_case, torch_x, x_local)

            device_mesh = dtorch.DeviceMesh(device, [0, 1])
            x_s0_from_x_local = x_local.redistribute(device_mesh=device_mesh, placements=[Shard(0)])
            assert_tensor_allclose(test_case, x_local, x_s0_from_x_local)

            x_distribute_to_local = x_s0_from_x_local.redistribute_like(x_local)
            assert_tensor_allclose(test_case, x_local, x_distribute_to_local)

            x_p = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[dtorch.Partial()],
                graph=graph,
            )
            x_r = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Replicate()],
                graph=graph,
            )
            x_s0 = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Shard(0)],
                graph=graph,
            )
            x_s1 = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Shard(-1)],
                graph=graph,
            )

            x_r_from_p = x_p.redistribute(device_mesh=device_mesh, placements=[Replicate()])
            assert_tensor_allclose(test_case, x_r, x_r_from_p)

            # x_s0_from_p = x_p.redistribute(
            #     device_mesh=device_mesh, placements=[Shard(0)]
            # )
            # assert_tensor_allclose(test_case, x_r, x_s0_from_p)

            x_r_from_s0 = x_s0.redistribute(device_mesh=device_mesh, placements=[Replicate()])
            assert_tensor_allclose(test_case, x_r, x_r_from_s0)
            test_case.assertTrue(x_r.dtype == x_r_from_s0.dtype)
            test_case.assertTrue(x_r.placements == x_r_from_s0.placements)

            x_r_from_s1 = x_s1.redistribute(device_mesh=device_mesh, placements=[Replicate()])
            assert_tensor_allclose(test_case, x_r, x_r_from_s1)

            x_s0_from_s1 = x_s1.redistribute(device_mesh=device_mesh, placements=[Shard(0)])
            assert_tensor_allclose(test_case, x_r, x_s0_from_s1)

            x_s1_from_s0 = x_s0.redistribute(device_mesh=device_mesh, placements=[Shard(1)])
            assert_tensor_allclose(test_case, x_r, x_s1_from_s0)

            x_s1_from_r = x_r.redistribute(device_mesh=device_mesh, placements=[Shard(1)])
            assert_tensor_allclose(test_case, x_r, x_s1_from_r)

            # # 2D
            # x_shape = [9, 18]
            # torch_x = torch.rand(*x_shape, device=device, dtype=torch.float32)
            # x_local = dtorch.tensor(torch_x, graph=graph)
            # device_mesh = init_device_mesh(device, (4, 2))
            # x_s0_s1_from_x_local = x_local.redistribute(
            #     device_mesh=device_mesh, placements=[Shard(0), Shard(1)]
            # )
            # assert_tensor_allclose(test_case, x_local, x_s0_s1_from_x_local)

            # x_s0_s1 = dtorch.tensor(
            #     torch_x,
            #     device_mesh=device_mesh,
            #     placements=[Shard(0), Shard(1)],
            #     graph=graph,
            # )
            # x_r_s1_from_x_s0_s1 = x_s0_s1.redistribute(
            #     device_mesh=device_mesh, placements=[Replicate(), Shard(1)]
            # )
            # assert_tensor_allclose(test_case, torch_x, x_r_s1_from_x_s0_s1)
            # x_s0_r_from_x_s0_s1 = x_s0_s1.redistribute(
            #     device_mesh=device_mesh, placements=[Shard(0), Replicate()]
            # )
            # assert_tensor_allclose(test_case, torch_x, x_r_s1_from_x_s0_s1)

            # x_r_r = dtorch.tensor(
            #     torch_x,
            #     device_mesh=device_mesh,
            #     placements=[Replicate(), Replicate()],
            #     graph=graph,
            # )
            # x_s1_r_from_r_r = x_r_r.redistribute(
            #     device_mesh=device_mesh, placements=[Shard(1), Replicate()]
            # )
            # assert_tensor_allclose(test_case, torch_x, x_s1_r_from_r_r)

            # x_s1_s1_from_r_r = x_r_r.redistribute(
            #     device_mesh=device_mesh, placements=[Shard(1), Shard(1)]
            # )
            # assert_tensor_allclose(test_case, torch_x, x_s1_r_from_r_r)

        def test_distributed_tensor_redistribute_imp(test_case, device, graph):
            # device_mesh with dimention size 1
            x_shape = [5, 9]
            torch_x = torch.rand(*x_shape, device=device, dtype=torch.float32)
            device_mesh = init_device_mesh(device, (1))
            x_1d = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                graph=graph,
            )
            device_mesh = init_device_mesh(device, (1, 1, 1))
            x_2d_from_1d = x_1d.redistribute(
                device_mesh=device_mesh,
                placements=[Replicate(), Replicate(), Replicate()],
            )
            assert_tensor_allclose(test_case, x_1d, x_2d_from_1d)

            # 1D
            x_shape = [4, 8]
            torch_x = torch.rand(*x_shape, device=device, dtype=torch.float32)
            x_local = dtorch.tensor(torch_x, graph=graph)
            assert_tensor_allclose(test_case, torch_x, x_local)

            device_mesh = dtorch.DeviceMesh(device, [0, 1])
            x_s0_from_x_local = x_local.redistribute(device_mesh=device_mesh, placements=[Shard(0)])
            assert_tensor_allclose(test_case, x_local, x_s0_from_x_local)

            x_distribute_to_local = x_s0_from_x_local.redistribute_like(x_local)
            assert_tensor_allclose(test_case, x_local, x_distribute_to_local)

            x_p = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[dtorch.Partial()],
                graph=graph,
            )
            x_r = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Replicate()],
                graph=graph,
            )
            x_s0 = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Shard(0)],
                graph=graph,
            )
            x_s1 = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Shard(-1)],
                graph=graph,
            )

            x_r_from_p = x_p.redistribute(device_mesh=device_mesh, placements=[Replicate()])
            assert_tensor_allclose(test_case, x_r, x_r_from_p)

            x_s0_from_p = x_p.redistribute(device_mesh=device_mesh, placements=[Shard(0)])
            assert_tensor_allclose(test_case, x_r, x_s0_from_p)

            x_r_from_s0 = x_s0.redistribute(device_mesh=device_mesh, placements=[Replicate()])
            assert_tensor_allclose(test_case, x_r, x_r_from_s0)
            test_case.assertTrue(x_r.dtype == x_r_from_s0.dtype)
            test_case.assertTrue(x_r.placements == x_r_from_s0.placements)

            x_r_from_s1 = x_s1.redistribute(device_mesh=device_mesh, placements=[Replicate()])
            assert_tensor_allclose(test_case, x_r, x_r_from_s1)

            x_s0_from_s1 = x_s1.redistribute(device_mesh=device_mesh, placements=[Shard(0)])
            assert_tensor_allclose(test_case, x_r, x_s0_from_s1)

            x_s1_from_s0 = x_s0.redistribute(device_mesh=device_mesh, placements=[Shard(1)])
            assert_tensor_allclose(test_case, x_r, x_s1_from_s0)

            x_s1_from_r = x_r.redistribute(device_mesh=device_mesh, placements=[Shard(1)])
            assert_tensor_allclose(test_case, x_r, x_s1_from_r)

            # 2D
            device_mesh = init_device_mesh(device, (4, 2))
            if not graph.satisfy(device_mesh):
                return
            x_shape = [4, 16]
            torch_x = torch.rand(*x_shape, device=device, dtype=torch.float32)
            x_local = dtorch.tensor(torch_x, graph=graph)
            x_s0_s1_from_x_local = x_local.redistribute(device_mesh=device_mesh, placements=[Shard(0), Shard(1)])
            assert_tensor_allclose(test_case, x_local, x_s0_s1_from_x_local)

            x_s0_s1 = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Shard(0), Shard(1)],
                graph=graph,
            )
            x_r_s1_from_x_s0_s1 = x_s0_s1.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(1)])
            assert_tensor_allclose(test_case, torch_x, x_r_s1_from_x_s0_s1)
            x_s0_r_from_x_s0_s1 = x_s0_s1.redistribute(device_mesh=device_mesh, placements=[Shard(0), Replicate()])
            assert_tensor_allclose(test_case, torch_x, x_r_s1_from_x_s0_s1)

            x_r_r = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Replicate(), Replicate()],
                graph=graph,
            )
            x_s1_r_from_r_r = x_r_r.redistribute(device_mesh=device_mesh, placements=[Shard(1), Replicate()])
            assert_tensor_allclose(test_case, torch_x, x_s1_r_from_r_r)

            x_s1_s1_from_r_r = x_r_r.redistribute(device_mesh=device_mesh, placements=[Shard(1), Shard(1)])
            assert_tensor_allclose(test_case, torch_x, x_s1_r_from_r_r)

            # 3D
            device_mesh = init_device_mesh(device, (2, 2, 2))
            if not graph.satisfy(device_mesh):
                return
            x_shape = [2, 16, 128, 128]
            torch_x = torch.rand(*x_shape, device=device, dtype=torch.float32)
            x_s0_s2_s2 = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Shard(0), Shard(2), Shard(2)],
                graph=graph,
            )
            assert_tensor_allclose(test_case, torch_x, x_s0_s2_s2)
            x_r_r_r = dtorch.tensor(
                torch_x,
                device_mesh=device_mesh,
                placements=[Replicate(), Replicate(), Replicate()],
                graph=graph,
            )
            assert_tensor_allclose(test_case, torch_x, x_r_r_r)

            x_r_r_r_from_s0_s2_s2 = x_s0_s2_s2.redistribute(
                device_mesh=device_mesh,
                placements=[Replicate(), Replicate(), Replicate()],
            )
            assert_tensor_allclose(test_case, torch_x, x_r_r_r_from_s0_s2_s2)

            x_s0_s2_s2_from_r_r_r = x_r_r_r.redistribute(
                device_mesh=device_mesh, placements=[Shard(0), Shard(2), Shard(2)]
            )
            assert_tensor_allclose(test_case, torch_x, x_s0_s2_s2_from_r_r_r)

        # dtensorInSameDevice is now a process-global option controlled by DTORCH_DTENSOR_IN_SAME_DEVICE.
        # A single process can only run in one mode, so build one graph and gate the CUDA case on the
        # active mode: same-device mode emulates multiple GPUs on one, otherwise real multi-GPU is needed.
        # To exercise both modes, run the suite once with DTORCH_DTENSOR_IN_SAME_DEVICE=1 and once unset.
        graph = dtorch.Graph()
        test_distributed_tensor_redistribute_imp(test_case, "cpu", graph)
        test_distributed_uneven_split_tensor_redistribute_imp(test_case, "cpu", graph)
        if torch.cuda.is_available() and (
            dtorch.GlobalOption.get_dtensor_in_same_device() or torch.cuda.device_count() >= 2
        ):
            test_distributed_tensor_redistribute_imp(test_case, "cuda", graph)
            test_distributed_uneven_split_tensor_redistribute_imp(test_case, "cuda", graph)

    def test_distributed_tensor_redistribute_with_dim_name(test_case):
        x_shape = [2, 4, 8]
        device_mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=["dp", "cp", "tp"])
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh)
        dtorch_x = dtorch_x.redistribute_by_dict(
            placements_dict={
                "dp": Shard(0),
                "cp": Shard(1),
                "tp": Shard(1),
            }
        )
        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        test_case.assertTrue(dtorch_x.placements[0] == Shard(0))
        test_case.assertTrue(dtorch_x.placements[1] == Shard(1))
        test_case.assertTrue(dtorch_x.placements[2] == Shard(1))

        x_shape = [2, 4, 8]
        device_mesh = init_device_mesh("cpu", (2, 1, 2), mesh_dim_names=["dp", "cp", "tp"])
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x, device_mesh=device_mesh)
        dtorch_x = dtorch_x.redistribute_by_dict(
            placements_dict={
                "dp": Shard(0),
                "cp": Shard(1),
                "tp": Shard(1),
            }
        )
        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        test_case.assertTrue(dtorch_x.placements[0] == Shard(0))
        test_case.assertTrue(dtorch_x.placements[1] == Replicate())
        test_case.assertTrue(dtorch_x.placements[2] == Shard(1))

    def test_device_mesh(test_case):
        device_mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=["dp", "cp", "tp"])
        unbind_device_mesh = device_mesh.unbind(dims=["cp"])
        unbind_device_mesh = [it.mesh for it in unbind_device_mesh]

        unbind_torch_device_mesh = device_mesh.mesh.unbind(1)

        assert_tensor_allclose(test_case, unbind_device_mesh, unbind_torch_device_mesh)

        device_mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=["dp", "cp", "tp"])
        unbind_device_mesh = device_mesh.unbind(dims=["cp", "tp"])
        unbind_device_mesh = [it.mesh for it in unbind_device_mesh]

        unbind_torch_device_mesh = device_mesh.mesh.unbind(2)
        unbind_torch_device_mesh = [it for its in unbind_torch_device_mesh for it in its.unbind(1)]

        assert_tensor_allclose(test_case, unbind_device_mesh, unbind_torch_device_mesh)


if __name__ == "__main__":
    unittest.main()
