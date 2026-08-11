"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_expand_same_rank(test_case, shape, expand_shape, device):
    """Test expand where output rank equals input rank (the case that triggered
    the bug: expand_op.cc used `>` instead of `>=` for axis count assertion).
    """
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch_in.expand(*expand_shape)
    dtorch_out = dtorch_in.expand(*expand_shape)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestExpand(unittest.TestCase):
    def test_expand_same_rank(test_case):
        """Covers the fix: outputShape.NumAxis() >= inputShape.NumAxis() (was `>`).

        Tests expand where output rank equals input rank — the original `>` assertion
        incorrectly rejected these cases.
        """
        # (shape, expand_shape) pairs — not all combinations are valid
        test_pairs = [
            ((1, 1, 64, 64), (2, 1, 64, 64)),  # 4D -> 4D: exact causal mask expand case
            ((1, 3, 4), (2, 3, 4)),  # 3D -> 3D
            ((1, 3), (5, 3)),  # 2D -> 2D
        ]
        for shape, expand_shape in test_pairs:
            for device in ["cpu", "cuda"]:
                _test_expand_same_rank(test_case, shape, expand_shape, device)

    def test_expand_more_rank(test_case):
        """Expand to more dimensions (the original test logic)."""
        for device in ["cpu", "cuda"]:
            _test_expand_same_rank(test_case, (1, 4, 5), (4, -1, -1), device)


if __name__ == "__main__":
    unittest.main()
