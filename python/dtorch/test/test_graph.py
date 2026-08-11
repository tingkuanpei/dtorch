"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import asyncio
import unittest

import torch

import dtorch
from dtorch import DeviceMesh, VoidFutureCollect
import dtorch.nn.functional as F
from dtorch.test.test_util import assert_tensor_allclose


class TestGraph(unittest.TestCase):
    def test_graph_option(test_case):
        graph_option = dtorch.GraphOption()
        test_case.assertTrue(graph_option.per_device_per_process is None)
        graph_option.per_device_per_process = True
        test_case.assertTrue(graph_option.per_device_per_process is True)

    def test_functional_use(test_case):
        torch_in = torch.rand(2, 4)
        torch_out = torch.nn.functional.relu(torch_in)

        # default eager graph
        dtorch_in = dtorch.Tensor(torch_in)
        dtorch_out = dtorch.nn.functional.relu(dtorch_in)
        dtorch.Graph.default_graph().sync()
        assert_tensor_allclose(test_case, torch_out, dtorch_out)
        test_case.assertTrue(dtorch_in.graph == dtorch.Graph.default_graph())

        # other eager graph
        graph = dtorch.Graph()
        dtorch_in = dtorch.Tensor(torch_in, graph=graph)
        dtorch_out = F.relu(dtorch_in)
        assert_tensor_allclose(test_case, torch_out, dtorch_out)
        test_case.assertTrue(graph != dtorch.Graph.default_graph())
        test_case.assertTrue(graph == dtorch_in.graph)

        test_case.assertTrue(graph.default_device_mesh == DeviceMesh("cpu"))
        with graph.device_mesh_guard("cuda"):
            with graph.dtype_guard(torch.float16):
                test_case.assertTrue(graph.default_device_mesh == DeviceMesh("cuda"))
                test_case.assertTrue(graph.default_dtype == torch.float16)
                dtorch_x = dtorch.zeros(1, 2, graph=graph)
                test_case.assertTrue(dtorch_x.device == torch.device("cuda:0"))
                test_case.assertTrue(dtorch_x.dtype == torch.float16)
                test_case.assertTrue(dtorch_x.graph == graph)
        test_case.assertTrue(graph.default_device_mesh == DeviceMesh("cpu"))
        test_case.assertTrue(graph.default_dtype == torch.float32)


class TestGraphSyncFuture(unittest.TestCase):
    def test_sync_basic(test_case):
        """Test sync() blocks until completion."""
        torch_in = torch.rand(2, 4)
        dtorch_in = dtorch.Tensor(torch_in)
        torch_out = torch.nn.functional.relu(torch_in)
        dtorch_out = dtorch.nn.functional.relu(dtorch_in)

        # Blocking sync should not raise
        dtorch_in.graph.sync()
        assert_tensor_allclose(test_case, torch_out, dtorch_out)

    def test_sync_future_wait(test_case):
        """Test sync_future() returns VoidFutureCollect and wait() blocks."""
        torch_in = torch.rand(2, 4)
        dtorch_in = dtorch.Tensor(torch_in)
        dtorch_out = dtorch.nn.functional.relu(dtorch_in)

        future = dtorch_in.graph.sync_future()
        test_case.assertIsInstance(future, VoidFutureCollect)

        # Wait should not hang
        future.wait()
        assert_tensor_allclose(test_case, torch.nn.functional.relu(torch_in), dtorch_out)

    def test_sync_future_is_ready(test_case):
        """Test VoidFutureCollect.is_ready() non-blocking check."""
        torch_in = torch.rand(3, 5)
        dtorch_in = dtorch.Tensor(torch_in)
        dtorch.nn.functional.relu(dtorch_in)

        future = dtorch_in.graph.sync_future()

        # Poll until ready
        import time

        for _ in range(1000):
            if future.is_ready():
                break
            time.sleep(0.001)
        test_case.assertTrue(future.is_ready(), "VoidFutureCollect should become ready within polling loop")

    def test_sync_consistency(test_case):
        """Test sync() and sync_future().wait() both work."""
        torch_in = torch.rand(3, 3)
        dtorch_in = dtorch.Tensor(torch_in)
        dtorch_out = dtorch.nn.functional.relu(dtorch_in)
        torch_out = torch.nn.functional.relu(torch_in)

        # Blocking sync
        dtorch_in.graph.sync()
        sync_result = dtorch_out.to_torch()

        # Future sync
        dtorch_in2 = dtorch.Tensor(torch_in)
        dtorch_out2 = dtorch.nn.functional.relu(dtorch_in2)
        future = dtorch_in2.graph.sync_future()
        future.wait()
        future_result = dtorch_out2.to_torch()

        assert_tensor_allclose(test_case, torch_out, sync_result)
        assert_tensor_allclose(test_case, torch_out, future_result)
        assert_tensor_allclose(test_case, sync_result, future_result)

    def test_sync_future_await(test_case):
        """Test VoidFutureCollect supports await."""

        async def await_sync():
            torch_in = torch.rand(2, 4)
            dtorch_in = dtorch.Tensor(torch_in)
            dtorch.nn.functional.relu(dtorch_in)

            future = dtorch_in.graph.sync_future()
            await future

        asyncio.run(await_sync())


if __name__ == "__main__":
    unittest.main()
