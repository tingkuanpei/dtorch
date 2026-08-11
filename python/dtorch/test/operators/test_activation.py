"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_activation(test_case, shape, device, dtype):
    torch_in = 10 * torch.rand(*shape, device=device, dtype=dtype)
    # torch_in.clone() avoid inplace=True modifying the reference tensor
    dtorch_in = dtorch.Tensor(torch_in.clone())
    device_mesh = dtorch.DeviceMesh(device, range(max(shape[0], 2)))
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    placements_shard = [dtorch.Shard(0)]
    placements_replicate = [dtorch.Replicate()]
    dtorch_d_s0 = dtorch.Tensor(torch_in.clone(), device_mesh=device_mesh, placements=placements_shard)
    dtorch_d_r = dtorch.Tensor(torch_in.clone(), device_mesh=device_mesh, placements=placements_replicate)

    # ---- relu ----
    torch_out = torch.relu(torch_in)
    dtorch_out = dtorch.relu(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    dtorch_out = dtorch.relu(dtorch_d_s0)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)
    dtorch_out = dtorch.relu(dtorch_d_r)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    # relu nn.Module (inplace)
    dtorch_m = dtorch.nn.ReLU(inplace=True)
    dtorch_out = dtorch_m(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # ---- sigmoid ----
    torch_out = torch.sigmoid(torch_in)
    dtorch_out = dtorch.sigmoid(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    dtorch_out = dtorch.sigmoid(dtorch_d_s0)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)
    dtorch_out = dtorch.sigmoid(dtorch_d_r)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    # sigmoid nn.Module
    dtorch_m = dtorch.nn.Sigmoid()
    dtorch_out = dtorch_m(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # ---- leaky_relu ----
    torch_out = torch.nn.functional.leaky_relu(torch_in, negative_slope=0.2)
    dtorch_out = dtorch.nn.functional.leaky_relu(dtorch_in, negative_slope=0.2)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    dtorch_out = dtorch.nn.functional.leaky_relu(dtorch_d_s0, negative_slope=0.2)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)
    dtorch_out = dtorch.nn.functional.leaky_relu(dtorch_d_r, negative_slope=0.2)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    # leaky_relu nn.Module
    dtorch_m = dtorch.nn.LeakyReLU(negative_slope=0.2)
    dtorch_out = dtorch_m(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # leaky_relu inplace
    dtorch_m_inplace = dtorch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
    dtorch_in_clone = dtorch.Tensor(torch_in.clone())
    dtorch_out = dtorch_m_inplace(dtorch_in_clone)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # ---- elu ----
    torch_out = torch.nn.functional.elu(torch_in, alpha=2.0)
    dtorch_out = dtorch.nn.functional.elu(dtorch_in, alpha=2.0)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    dtorch_out = dtorch.nn.functional.elu(dtorch_d_s0, alpha=2.0)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)
    dtorch_out = dtorch.nn.functional.elu(dtorch_d_r, alpha=2.0)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    # elu nn.Module
    dtorch_m = dtorch.nn.ELU(alpha=2.0, inplace=True)
    dtorch_out = dtorch_m(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # ---- gelu ----
    torch_out = torch.nn.functional.gelu(torch_in)
    dtorch_out = dtorch.nn.functional.gelu(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    dtorch_out = dtorch.nn.functional.gelu(dtorch_d_s0)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)
    dtorch_out = dtorch.nn.functional.gelu(dtorch_d_r)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    # gelu nn.Module
    dtorch_m = dtorch.nn.GELU()
    dtorch_out = dtorch_m(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # gelu approximate="tanh"
    torch_out = torch.nn.functional.gelu(torch_in, approximate="tanh")
    dtorch_out = dtorch.nn.functional.gelu(dtorch_in, approximate="tanh")
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # gelu approximate="none" (explicit)
    torch_out = torch.nn.functional.gelu(torch_in, approximate="none")
    dtorch_out = dtorch.nn.functional.gelu(dtorch_in, approximate="none")
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # ---- silu ----
    torch_out = torch.nn.functional.silu(torch_in)
    dtorch_out = dtorch.nn.functional.silu(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    dtorch_out = dtorch.nn.functional.silu(dtorch_d_s0)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)
    dtorch_out = dtorch.nn.functional.silu(dtorch_d_r)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    # silu nn.Module
    dtorch_m = dtorch.nn.SiLU(inplace=True)
    dtorch_out = dtorch_m(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_activation_scalar(test_case, device, dtype):
    """Element-wise activations on scalar tensors (Shard not applicable, Replicate only)."""
    torch_in = 10 * torch.rand((1,), device=device, dtype=dtype).squeeze()
    dtorch_in = dtorch.Tensor(torch_in.clone())

    torch_out = torch.relu(torch_in)
    dtorch_out = dtorch.relu(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.sigmoid(torch_in)
    dtorch_out = dtorch.sigmoid(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.leaky_relu(torch_in, negative_slope=0.2)
    dtorch_out = dtorch.nn.functional.leaky_relu(dtorch_in, negative_slope=0.2)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.elu(torch_in, alpha=2.0)
    dtorch_out = dtorch.nn.functional.elu(dtorch_in, alpha=2.0)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.gelu(torch_in)
    dtorch_out = dtorch.nn.functional.gelu(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.silu(torch_in)
    dtorch_out = dtorch.nn.functional.silu(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_activation_int_relu(test_case, device):
    """ReLU on integer tensor should work (only ReLU supports integer inputs)."""
    torch_in = torch.randint(-5, 5, (3, 4), device=device)
    dtorch_in = dtorch.Tensor(torch_in.clone())

    torch_out = torch.relu(torch_in)
    dtorch_out = dtorch.relu(dtorch_in)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_activation_partial_rejected(test_case, device):
    """Non-linear activations should reject Partial() input."""
    shape = (4, 8)
    torch_in = torch.rand(*shape, device=device)

    # Partial placements — e.g., from a matmul before all-reduce
    partial_placements = [dtorch.Partial()]

    # Use a 1D device mesh with a Partial placement on the only axis
    device_mesh = dtorch.DeviceMesh(device, range(4))
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_d_partial = dtorch.Tensor(torch_in.clone(), device_mesh=device_mesh, placements=partial_placements)

    # All activation functions should reject Partial input
    with test_case.assertRaises(ValueError):
        dtorch.relu(dtorch_d_partial)

    with test_case.assertRaises(ValueError):
        dtorch.sigmoid(dtorch_d_partial)

    with test_case.assertRaises(ValueError):
        dtorch.nn.functional.leaky_relu(dtorch_d_partial)

    with test_case.assertRaises(ValueError):
        dtorch.nn.functional.elu(dtorch_d_partial)

    with test_case.assertRaises(ValueError):
        dtorch.nn.functional.gelu(dtorch_d_partial)

    with test_case.assertRaises(ValueError):
        dtorch.nn.functional.silu(dtorch_d_partial)


def _test_gelu_invalid_approximate(test_case):
    """GELU with invalid approximate should raise."""
    torch_in = torch.rand(2, 3)
    dtorch_in = dtorch.Tensor(torch_in)

    with test_case.assertRaises(ValueError):
        dtorch.nn.functional.gelu(dtorch_in, approximate="invalid_value")


class TestActivation(unittest.TestCase):
    def test_activation(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(3,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32, torch.float16]
        for arg in gen_arg_list(arg_dict):
            _test_activation(test_case, *arg)

    def test_activation_scalar(test_case):
        for device in ["cpu", "cuda"]:
            for dtype in [torch.float32, torch.float16]:
                _test_activation_scalar(test_case, device, dtype)

    def test_activation_int_relu(test_case):
        for device in ["cpu", "cuda"]:
            _test_activation_int_relu(test_case, device)

    def test_activation_partial_rejected(test_case):
        for device in ["cpu", "cuda"]:
            _test_activation_partial_rejected(test_case, device)

    def test_gelu_invalid_approximate(test_case):
        _test_gelu_invalid_approximate(test_case)


if __name__ == "__main__":
    unittest.main()
