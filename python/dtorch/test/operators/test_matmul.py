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


# ---------------------------------------------------------------------------
# Non-distributed tests: DTorch matmul vs PyTorch matmul
# ---------------------------------------------------------------------------


def _test_matmul(test_case, shape_a, shape_b, device, dtype):
    """Generic matmul test: DTorch output must match PyTorch output."""
    torch_a = torch.rand(*shape_a, dtype=dtype, device=device)
    torch_b = torch.rand(*shape_b, dtype=dtype, device=device)
    dtorch_a = dtorch.Tensor(torch_a)
    dtorch_b = dtorch.Tensor(torch_b)

    torch_out = torch.matmul(torch_a, torch_b)
    dtorch_out = dtorch.matmul(dtorch_a, dtorch_b)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


# ---------------------------------------------------------------------------
# Distributed tests: DTorch matmul with DTensor Shard / Replicate / Partial placements
#
# IMPORTANT: PlacementSignature rules must align with local tensor shape
# compatibility. For matmul, the contracted (K) dimension must match locally:
#   - Both A and B sharded on K → each device has (M, K/n) and (K/n, N) → OK
#   - Both A and B replicated on K → each device has (M, K) and (K, N) → OK
#   - Only one K sharded → K/n ≠ K → local shape mismatch → NOT supported
# ---------------------------------------------------------------------------


def _test_distributed_matmul(test_case, shape_a, shape_b, placements_a, placements_b, device):
    """Generic distributed matmul test with given input placements.

    The mesh size is determined from the sharded dimension:
    - For Shard(dim) on A: mesh_size = shape_a[dim]
    - For Shard(dim) on B: mesh_size = shape_b[dim]
    - Otherwise uses the contracted (K) dimension.
    """
    torch_a = torch.rand(*shape_a, dtype=torch.float32, device=device)
    torch_b = torch.rand(*shape_b, dtype=torch.float32, device=device)

    # Determine mesh size: use the sharded dim size so mesh divides cleanly
    mesh_size = None
    for pl in placements_a:
        if pl.is_shard():
            mesh_size = shape_a[pl.get_shard_index()]
    if mesh_size is None:
        for pl in placements_b:
            if pl.is_shard():
                mesh_size = shape_b[pl.get_shard_index()]
    if mesh_size is None:
        # All replicated: use K dim size (the contracted dimension)
        mesh_size = shape_a[-1]  # K dim

    device_mesh = dtorch.DeviceMesh(device, range(mesh_size), mesh_dim_names=["tp"])
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    dtorch_a = dtorch.Tensor(torch_a, device_mesh=device_mesh, placements=placements_a)
    dtorch_b = dtorch.Tensor(torch_b, device_mesh=device_mesh, placements=placements_b)

    torch_out = torch.matmul(torch_a, torch_b)
    dtorch_out = dtorch.matmul(dtorch_a, dtorch_b)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


# ---------------------------------------------------------------------------
# Non-distributed test class
# ---------------------------------------------------------------------------


class TestMatmulNonDistributed(unittest.TestCase):
    """Non-distributed matmul tests covering all torch.matmul dimension combinations."""

    def test_matmul_1d_1d(self):
        """Dot product: 1D × 1D → scalar."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((8,), (8,)),
            ((16,), (16,)),
            ((32,), (32,)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_2d_2d(self):
        """Matrix-matrix: 2D × 2D → 2D."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((8, 16), (16, 4)),
            ((3, 4), (4, 5)),
            ((5, 6), (6, 3)),
            ((1, 10), (10, 1)),
            ((64, 128), (128, 32)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_1d_2d(self):
        """Vector-matrix: 1D × 2D → 1D."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((16,), (16, 4)),
            ((4,), (4, 5)),
            ((6,), (6, 3)),
            ((1,), (1, 10)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_2d_1d(self):
        """Matrix-vector: 2D × 1D → 1D."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((8, 16), (16,)),
            ((3, 4), (4,)),
            ((5, 6), (6,)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_batched_batched(self):
        """Batched matmul with matching batch dims: 3D × 3D → 3D."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((10, 3, 4), (10, 4, 5)),
            ((2, 8, 16), (2, 16, 4)),
            ((1, 5, 6), (1, 6, 3)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_batched_broadcasted(self):
        """Batched matmul with broadcast: broadcast batch dims between inputs."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((10, 3, 4), (4,)),  # 3D × 1D
            ((3,), (10, 3, 4)),  # 1D × 3D
            ((10, 3, 4), (4, 5)),  # 3D × 2D (broadcast batch)
            ((2, 1, 4, 6), (1, 3, 6, 3)),  # 4D batched: batch [2,1] vs [1,3] → [2,3] ✓
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_nd_nd(self):
        """Higher-dimensional batched matmul: 4D × 4D → 4D."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((2, 3, 8, 16), (2, 3, 16, 4)),
            ((1, 4, 5, 6), (1, 4, 6, 3)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_1d_nd(self):
        """Vector × ND tensor: 1D × 3D/4D."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((16,), (10, 16, 4)),  # 1D × 3D (batched: 1D × (batch, K, N))
            ((6,), (2, 3, 6, 3)),  # 1D × 4D
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_nd_1d(self):
        """ND tensor × vector: 3D/4D × 1D."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((10, 8, 16), (16,)),  # 3D × 1D
            ((2, 3, 5, 6), (6,)),  # 4D × 1D
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)

    def test_matmul_different_dtypes(self):
        """Matmul with different dtypes: float16, float32."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((8, 16), (16, 4)),
            ((3, 4), (4, 5)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float16, torch.float32]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device, dtype = arg
            _test_matmul(self, shape_a, shape_b, device, dtype)


# ---------------------------------------------------------------------------
# Distributed test class
#
# Valid distributed configurations for matmul (2D×2D GEMM as example):
#   |  A placement  |  B placement  |  Y placement  | Local shapes valid? |
#   |---------------|---------------|---------------|---------------------|
#   | Shard(M)      | Replicate     | Shard(M)      | ✓ A[M/n,K] × B[K,N] |
#   | Replicate     | Shard(N)      | Shard(N)      | ✓ A[M,K] × B[K,N/n] |
#   | Shard(K)      | Shard(K)      | Partial       | ✓ A[M,K/n]×B[K/n,N] |
#   | Replicate     | Replicate     | Replicate     | ✓ (default fallback) |
#
# One-sided K shard (e.g. Shard(K_A)+R or R+Shard(K_B)) causes local K-dim
# mismatch (K/n vs K) — the current framework does NOT auto-redistribute
# for this case, so these are excluded from tests.
# ---------------------------------------------------------------------------


class TestMatmulDistributed(unittest.TestCase):
    """Distributed matmul tests with DTensor Shard/Replicate/Partial placements."""

    # --- 2D × 2D distributed tests ---

    def test_matmul_2d_2d_shard_m(self):
        """2D×2D GEMM: Shard A on M dim, B replicated → output Shard on M."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8, 6)),
            ((8, 16), (16, 4)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(0)],
                placements_b=[dtorch.Replicate()],
                device=device,
            )

    def test_matmul_2d_2d_shard_n(self):
        """2D×2D GEMM: A replicated, B Shard on N dim → output Shard on N."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8, 6)),
            ((8, 16), (16, 4)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Replicate()],
                placements_b=[dtorch.Shard(1)],
                device=device,
            )

    def test_matmul_2d_2d_shard_k_both(self):
        """2D×2D GEMM: Shard A on K dim + Shard B on K dim → Partial output
        (requires all-reduce). Local shapes: A[M, K/n] × B[K/n, N] → valid."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8, 6)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(1)],
                placements_b=[dtorch.Shard(0)],
                device=device,
            )

    # --- 1D × 2D distributed tests ---

    def test_matmul_1d_2d_shard_both_k(self):
        """1D×2D: Shard both A and B on K dim → Partial output.
        Local shapes: A[K/n] × B[K/n, N] → compatible."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((8,), (8, 6)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(0)],
                placements_b=[dtorch.Shard(0)],
                device=device,
            )

    def test_matmul_1d_2d_shard_n(self):
        """1D×2D: Shard B on N dim → output Shard on N.
        Local shapes: A[K] × B[K, N/n] → compatible."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((8,), (8, 6)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Replicate()],
                placements_b=[dtorch.Shard(1)],
                device=device,
            )

    # --- 2D × 1D distributed tests ---

    def test_matmul_2d_1d_shard_both_k(self):
        """2D×1D: Shard both A and B on K dim → Partial output.
        Local shapes: A[M, K/n] × B[K/n] → compatible."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8,)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(1)],
                placements_b=[dtorch.Shard(0)],
                device=device,
            )

    def test_matmul_2d_1d_shard_m(self):
        """2D×1D: Shard A on M dim → output Shard on M.
        Local shapes: A[M/n, K] × B[K] → compatible."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8,)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(0)],
                placements_b=[dtorch.Replicate()],
                device=device,
            )

    # --- Batched (3D) distributed tests ---

    def test_matmul_batched_shard_m(self):
        """Batched 3D×3D GEMM: Shard A on M dim (dim 1), B replicated → output Shard on M.
        Local shapes: A[B, M/n, K] × B[B, K, N] → compatible."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((2, 4, 8), (2, 8, 6)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(1)],
                placements_b=[dtorch.Replicate()],
                device=device,
            )

    def test_matmul_batched_shard_n(self):
        """Batched 3D×3D GEMM: A replicated, B Shard on N dim (dim 2) → output Shard on N.
        Local shapes: A[B, M, K] × B[B, K, N/n] → compatible."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((2, 4, 8), (2, 8, 6)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Replicate()],
                placements_b=[dtorch.Shard(2)],
                device=device,
            )

    def test_matmul_batched_shard_k_both(self):
        """Batched 3D×3D GEMM: Shard both A and B on K dim → Partial output.
        Local shapes: A[B, M, K/n] × B[B, K/n, N] → compatible, Partial result."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((2, 4, 8), (2, 8, 6)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(2)],
                placements_b=[dtorch.Shard(1)],
                device=device,
            )

    # --- Broadcast batched distributed tests ---

    def test_matmul_broadcast_batched_shard_batch(self):
        """Broadcast batched: 3D×2D, shard on batch dim of A.
        Local shapes: A[B/n, M, K] × B[K, N] (with broadcast batch B/n→B) → compatible."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8, 16), (16, 4)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_distributed_matmul(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(0)],
                placements_b=[dtorch.Replicate()],
                device=device,
            )

    # --- Multi-mesh-dimension distributed tests ---

    def test_matmul_multi_mesh_shard_m_n(self):
        """2D×2D with 2D mesh: Shard M and N across different mesh dims.
        Local shapes: A[M/2, K] × B[K, N/2] → compatible across both mesh dims."""
        torch_a = torch.rand(4, 8, dtype=torch.float32, device="cpu")
        torch_b = torch.rand(8, 6, dtype=torch.float32, device="cpu")

        device_mesh = dtorch.DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=["dp", "tp"])
        if not dtorch.default_graph.satisfy(device_mesh):
            return

        dtorch_a = dtorch.Tensor(torch_a, device_mesh=device_mesh, placements=[dtorch.Shard(0), dtorch.Replicate()])
        dtorch_b = dtorch.Tensor(torch_b, device_mesh=device_mesh, placements=[dtorch.Replicate(), dtorch.Shard(1)])

        torch_out = torch.matmul(torch_a, torch_b)
        dtorch_out = dtorch.matmul(dtorch_a, dtorch_b)
        assert_tensor_allclose(self, torch_out, dtorch_out)

    # --- Partial placement tests ---
    #
    # Partial output is produced when:
    #   - Both inputs are Shard on the contracted K dim (K_A Shard + K_B Shard → Partial)
    #   - One input has Partial placement (Partial + R → Partial, R + Partial → Partial)
    #   - Edge cases: Shard(M) + Shard(K_B) → Partial, Shard(K_A) + Shard(N) → Partial
    #
    # The output Partial tensor must be all-reduced to match the reference.
    # assert_tensor_allclose triggers materialization (to_torch) which performs
    # the implicit all-reduce.


def _test_partial_output_placement(test_case, shape_a, shape_b, placements_a, placements_b, device):
    """Verify output placement type matches expected placement."""
    torch_a = torch.rand(*shape_a, dtype=torch.float32, device=device)
    torch_b = torch.rand(*shape_b, dtype=torch.float32, device=device)

    mesh_size = shape_a[-1]  # K dim
    device_mesh = dtorch.DeviceMesh(device, range(mesh_size), mesh_dim_names=["tp"])
    if not dtorch.default_graph.satisfy(device_mesh):
        return

    dtorch_a = dtorch.Tensor(torch_a, device_mesh=device_mesh, placements=placements_a)
    dtorch_b = dtorch.Tensor(torch_b, device_mesh=device_mesh, placements=placements_b)

    torch_out = torch.matmul(torch_a, torch_b)
    dtorch_out = dtorch.matmul(dtorch_a, dtorch_b)

    # Verify values match (all-reduce happens implicitly on materialization)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)

    # Verify output has Partial placement
    out_placements = dtorch_out.placements
    has_partial = any(p.is_partial() for p in out_placements)
    test_case.assertTrue(has_partial, f"Expected Partial in output placements, got {out_placements}")

    def test_matmul_2d_2d_partial_output_from_k_shard(self):
        """2D×2D GEMM: Shard both K dims → Partial output, verify placement."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8, 6)),
            ((8, 16), (16, 4)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_partial_output_placement(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(1)],
                placements_b=[dtorch.Shard(0)],
                device=device,
            )

    # --- Edge case Partial tests (2D×2D) ---

    def test_matmul_2d_2d_edge_shard_m_plus_kb(self):
        """2D×2D: Shard(M) on A + Shard(K) on B → Partial (edge case rule).
        Even though M is sharded on A, the K shard on B forces Partial output
        because the contracted dimension is partially computed on each device."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8, 6)),
            ((8, 16), (16, 4)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_partial_output_placement(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(0)],  # Shard(M) on A
                placements_b=[dtorch.Shard(0)],  # Shard(K) on B
                device=device,
            )

    def test_matmul_2d_2d_edge_shard_ka_plus_n(self):
        """2D×2D: Shard(K) on A + Shard(N) on B → Partial (edge case rule).
        The K shard on A forces Partial, overriding the normal N-dim Shard propagation."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8, 6)),
            ((8, 16), (16, 4)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_partial_output_placement(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(1)],  # Shard(K) on A
                placements_b=[dtorch.Shard(1)],  # Shard(N) on B
                device=device,
            )

    # --- Edge case Partial tests (1D×2D and 2D×1D) ---

    def test_matmul_1d_2d_edge_shard_ka_plus_n(self):
        """1D×2D: Shard(K) on A + Shard(N) on B → Partial (edge case rule).
        The K shard forces Partial output even when N is also sharded on B."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((8,), (8, 6)),
            ((16,), (16, 4)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_partial_output_placement(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(0)],  # Shard(K) on A
                placements_b=[dtorch.Shard(1)],  # Shard(N) on B
                device=device,
            )

    def test_matmul_2d_1d_edge_shard_m_plus_kb(self):
        """2D×1D: Shard(M) on A + Shard(K) on B → Partial (edge case rule).
        The K shard on B forces Partial output, overriding M-dim Shard propagation."""
        arg_dict = OrderedDict()
        arg_dict["shape_pair"] = [
            ((4, 8), (8,)),
            ((8, 16), (16,)),
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            (shape_a, shape_b), device = arg
            _test_partial_output_placement(
                self,
                shape_a,
                shape_b,
                placements_a=[dtorch.Shard(0)],  # Shard(M) on A
                placements_b=[dtorch.Shard(0)],  # Shard(K) on B (only dim)
                device=device,
            )

    # --- Partial input propagation tests ---

    def test_matmul_partial_input_a(self):
        """Partial input A × Replicate B → Partial output.
        Creates Partial A via K-shard matmul, then uses it in a second matmul."""
        torch_a = torch.rand(4, 8, dtype=torch.float32, device="cpu")
        torch_b = torch.rand(8, 6, dtype=torch.float32, device="cpu")
        torch_c = torch.rand(6, 3, dtype=torch.float32, device="cpu")

        device_mesh = dtorch.DeviceMesh("cpu", range(8), mesh_dim_names=["tp"])
        if not dtorch.default_graph.satisfy(device_mesh):
            return

        # Step 1: matmul with K-shard → Partial output
        dtorch_a = dtorch.Tensor(torch_a, device_mesh=device_mesh, placements=[dtorch.Shard(1)])
        dtorch_b = dtorch.Tensor(torch_b, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
        dtorch_partial = dtorch.matmul(dtorch_a, dtorch_b)

        # The output should have Partial placement
        self.assertTrue(any(p.is_partial() for p in dtorch_partial.placements))

        # Step 2: Partial input × Replicate → Partial output
        dtorch_c = dtorch.Tensor(torch_c, device_mesh=device_mesh, placements=[dtorch.Replicate()])
        dtorch_out = dtorch.matmul(dtorch_partial, dtorch_c)

        # Verify the chained result
        torch_partial = torch.matmul(torch_a, torch_b)
        torch_out = torch.matmul(torch_partial, torch_c)
        assert_tensor_allclose(self, torch_out, dtorch_out)

    def test_matmul_partial_input_b(self):
        """Replicate A × Partial input B → Partial output.
        Creates Partial B via K-shard matmul, then uses it in a second matmul."""
        torch_a = torch.rand(4, 8, dtype=torch.float32, device="cpu")
        torch_b = torch.rand(8, 6, dtype=torch.float32, device="cpu")
        torch_c = torch.rand(6, 4, dtype=torch.float32, device="cpu")

        device_mesh = dtorch.DeviceMesh("cpu", range(8), mesh_dim_names=["tp"])
        if not dtorch.default_graph.satisfy(device_mesh):
            return

        # Step 1: matmul with K-shard → Partial B
        dtorch_a = dtorch.Tensor(torch_a, device_mesh=device_mesh, placements=[dtorch.Shard(1)])
        dtorch_b = dtorch.Tensor(torch_b, device_mesh=device_mesh, placements=[dtorch.Shard(0)])
        dtorch_partial = dtorch.matmul(dtorch_a, dtorch_b)

        self.assertTrue(any(p.is_partial() for p in dtorch_partial.placements))

        # Step 2: Replicate A × Partial B → Partial output
        dtorch_c = dtorch.Tensor(torch_c, device_mesh=device_mesh, placements=[dtorch.Replicate()])
        dtorch_out = dtorch.matmul(dtorch_c, dtorch_partial)

        # Verify the chained result
        torch_partial = torch.matmul(torch_a, torch_b)
        torch_out = torch.matmul(torch_c, torch_partial)
        assert_tensor_allclose(self, torch_out, dtorch_out)


if __name__ == "__main__":
    unittest.main()
