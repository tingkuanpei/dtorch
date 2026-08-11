"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
import time
import torch

import dtorch
from dtorch.test.test_util import assert_tensor_allclose
from dtorch.graph import GraphOption
from dtorch.nn.parameter import Parameter


class TestTensor(unittest.TestCase):
    def test_tensor_constructor(test_case):
        dtorch_x = dtorch.Tensor()
        torch_x = torch.Tensor()
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        x_shape = [2, 3, 4, 1]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)
        dtorch_x = dtorch.tensor(torch_x)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        torch_x = torch_x.contiguous()
        dtorch_x = dtorch_x.contiguous()
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        dtorch_x = dtorch.Tensor(1, 2)
        test_case.assertTrue(dtorch_x.shape == (1, 2))

        dtorch_x = dtorch.Tensor(torch.Size((1, 2)))
        test_case.assertTrue(dtorch_x.dtype == torch.float32)
        test_case.assertTrue(dtorch_x.device == torch.device("cpu"))
        test_case.assertTrue(dtorch_x.shape == (1, 2))

        torch_a = torch.rand(*x_shape, dtype=torch.float32)
        torch_b = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_a = dtorch.Tensor(torch_a)
        dtorch_b = dtorch.Tensor(torch_b)
        dtorch_b.copy_(dtorch_a)
        assert_tensor_allclose(test_case, dtorch_a, dtorch_b)

        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)
        test_case.assertTrue(dtorch_x.dtype == torch.float32)
        test_case.assertTrue(dtorch_x.device == torch.device("cpu"))

        if torch.cuda.is_available():
            torch_x = torch_x.to(device="cuda:0", dtype=torch.int32)
            dtorch_x = dtorch_x.to(device="cuda:0", dtype=torch.int32)
            test_case.assertTrue(dtorch_x.dtype == torch.int32)
            test_case.assertTrue(dtorch_x.device == torch.device("cuda:0"))
            assert_tensor_allclose(test_case, torch_x, dtorch_x)

        dtorch_x = dtorch_x.to("cpu")
        test_case.assertTrue(dtorch_x.device == torch.device("cpu"))
        dtorch_x = dtorch_x.to(torch.float)
        test_case.assertTrue(dtorch_x.dtype == torch.float)

        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)
        torch_x = torch_x.half()
        dtorch_x = dtorch_x.half()
        test_case.assertTrue(torch_x.dtype == dtorch_x.dtype)
        assert_tensor_allclose(test_case, torch_x, dtorch_x)

        test_case.assertTrue(torch.finfo(torch_x.dtype).max == dtorch.finfo(dtorch_x.dtype).max)

    def test_tensor_constructor_time(test_case):
        def test_tensor_constructor_time_imp(test_case, device):
            x_shape = [1024, 1024, 4]
            torch_x = torch.rand(*x_shape, device=device, dtype=torch.float32)

            # per_device_per_process=True
            process_graph = dtorch.Graph(GraphOption(per_device_per_process=True))
            dtorch_x = dtorch.Tensor(torch_x, graph=process_graph)
            process_graph.sync()
            time_start = time.time()

            dtorch_x = dtorch.Tensor(torch_x, graph=process_graph)
            assert_tensor_allclose(test_case, torch_x, dtorch_x)

            process_graph.sync()
            time_end = time.time()
            process_time_duration = time_end - time_start

            # per_device_per_process=False
            thread_graph = dtorch.Graph(GraphOption(per_device_per_process=False))
            dtorch_x = dtorch.Tensor(torch_x, graph=thread_graph)
            thread_graph.sync()
            time_start = time.time()

            dtorch_x = dtorch.Tensor(torch_x, graph=thread_graph)
            assert_tensor_allclose(test_case, torch_x, dtorch_x)

            thread_graph.sync()
            time_end = time.time()
            thread_time_duration = time_end - time_start

            # check tensor constructor time
            test_case.assertTrue(
                process_time_duration < 2 * thread_time_duration + 0.2,
                f"device: {device}, process_time_duration: {process_time_duration}, thread_time_duration: {thread_time_duration}",
            )

        test_tensor_constructor_time_imp(test_case, "cpu")
        if torch.cuda.is_available():
            test_tensor_constructor_time_imp(test_case, "cuda:0")

    def test_tensor_nozero(test_case):
        x_shape = [8, 4]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)

        torch_y = torch_x.nonzero()
        dtorch_y = dtorch_x.nonzero()
        assert_tensor_allclose(test_case, torch_y, dtorch_y)

    def test_size(test_case):
        """Test tensor.size() and tensor.size(dim) with negative index support."""
        x_shape = [2, 3, 4, 5]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)

        # size() with no arguments — should return torch.Size matching PyTorch
        test_case.assertEqual(torch_x.size(), dtorch_x.size())
        test_case.assertEqual(tuple(torch_x.size()), tuple(dtorch_x.size()))

        # size(dim) — positive indices
        for dim in range(torch_x.ndim):
            test_case.assertEqual(torch_x.size(dim), dtorch_x.size(dim), f"Mismatch at dim={dim}")

        # size(dim) — negative indices
        for dim in range(-torch_x.ndim, 0):
            test_case.assertEqual(torch_x.size(dim), dtorch_x.size(dim), f"Mismatch at dim={dim}")

        # Convenience patterns used in real code: mask.size(-1)
        test_case.assertEqual(torch_x.size(-1), dtorch_x.size(-1))
        test_case.assertEqual(torch_x.size(-2), dtorch_x.size(-2))

    def test_param_data_setter(test_case):
        """Test param.data = param.data.to(torch.float32) pattern used in transformers modeling_utils.py#L5163."""

        # --- torch reference ---
        torch_t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        test_case.assertEqual(torch_t.data.dtype, torch.float16)

        torch_p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))
        torch_p_id_before = id(torch_p)
        torch_p.data = torch_p.data.to(torch.float32)
        test_case.assertEqual(id(torch_p), torch_p_id_before)
        test_case.assertEqual(torch_p.dtype, torch.float32)

        # --- dtorch --- (must match torch behavior above)
        t = dtorch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        test_case.assertEqual(t.data.dtype, torch.float16)

        p = Parameter(dtorch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))
        p_id_before = id(p)
        p.data = p.data.to(torch.float32)
        test_case.assertEqual(id(p), p_id_before)
        test_case.assertEqual(p.dtype, torch.float32)


class TestTensorFuture(unittest.TestCase):
    def test_get_basic(test_case):
        """Test to_torch_async() returns TensorFuture and get() returns correct tensor."""
        x_shape = [2, 3, 4]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)

        future = dtorch_x.to_torch_async()
        test_case.assertIsInstance(future, dtorch.TensorFuture)

        result = future.get()
        assert_tensor_allclose(test_case, torch_x, result)

    def test_is_ready(test_case):
        """Test TensorFuture.is_ready() non-blocking value readiness check."""
        x_shape = [3, 5]
        torch_x = torch.ones(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)

        future = dtorch_x.to_torch_async()

        # Poll using is_ready() until the value is available
        for _ in range(1000):
            if future.is_ready():
                break
            time.sleep(0.001)
        test_case.assertTrue(future.is_ready(), "Future should become ready within polling loop")

        result = future.get()
        assert_tensor_allclose(test_case, torch_x, result)

    def test_wait(test_case):
        """Test TensorFuture.wait() with timeout."""
        x_shape = [4, 4]
        torch_x = torch.zeros(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)

        future = dtorch_x.to_torch_async()

        # Wait with sufficient timeout (5000 ms)
        test_case.assertTrue(future.wait_for(5000))
        result = future.get()
        assert_tensor_allclose(test_case, torch_x, result)

    def test_consistency_with_sync(test_case):
        """Test sync to_torch() and async to_torch_async().get() return same result."""
        x_shape = [3, 3]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)

        sync_result = dtorch_x.to_torch()
        async_result = dtorch_x.to_torch_async().get()

        assert_tensor_allclose(test_case, torch_x, sync_result)
        assert_tensor_allclose(test_case, sync_result, async_result)

    def test_multiple_gets(test_case):
        """Test multiple async gets on different tensors."""
        x_shape = [2, 4]
        torch_a = torch.rand(*x_shape, dtype=torch.float32)
        torch_b = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_a = dtorch.Tensor(torch_a)
        dtorch_b = dtorch.Tensor(torch_b)

        future_a = dtorch_a.to_torch_async()
        future_b = dtorch_b.to_torch_async()

        result_a = future_a.get()
        result_b = future_b.get()

        assert_tensor_allclose(test_case, torch_a, result_a)
        assert_tensor_allclose(test_case, torch_b, result_b)

    def test_large_tensor(test_case):
        """Test async get with a larger tensor."""
        x_shape = [128, 256]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)

        future = dtorch_x.to_torch_async()
        result = future.get()

        assert_tensor_allclose(test_case, torch_x, result)

    def test_await(test_case):
        """Test asyncio await on TensorFuture."""
        import asyncio

        x_shape = [2, 3]
        torch_x = torch.ones(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)

        async def async_get(tensor):
            future = tensor.to_torch_async()
            result = await future
            return result

        result = asyncio.run(async_get(dtorch_x))
        assert_tensor_allclose(test_case, torch_x, result)


if __name__ == "__main__":
    unittest.main()
