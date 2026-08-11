"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest

import torch

import dtorch
from dtorch.test.test_util import assert_tensor_equal


class TestGetItem(unittest.TestCase):
    def test_tensor_index(test_case):
        x_shape = [8, 7, 4, 6, 10]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_y = torch_x[:, :, 0]
        dtorch_y = dtorch_x[:, :, 0]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        torch_y = torch_x[2:3, ..., -2]
        dtorch_y = dtorch_x[2:3, ..., -2]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        torch_y = torch_x[2:3, :, None, ..., -2, None, ...]
        dtorch_y = dtorch_x[2:3, :, None, ..., -2, None, ...]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        for a, b in zip(torch_x, dtorch_x):
            assert_tensor_equal(test_case, a, b)

    def test_tensor_integer_array_indexing(test_case):
        """Test advanced indexing with integer tensor arrays."""
        x_shape = [8, 7, 4, 6, 10]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        # Case 1: single tensor index
        idx1 = torch.tensor([0, 1, 2], dtype=torch.int64)
        dtorch_idx1 = dtorch.Tensor(idx1)
        torch_y = torch_x[idx1]
        dtorch_y = dtorch_x[dtorch_idx1]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 2: two tensor indices (consecutive, same shape)
        idx2 = torch.tensor([1, 3, 5], dtype=torch.int64)
        dtorch_idx2 = dtorch.Tensor(idx2)
        torch_y = torch_x[idx1, idx2]
        dtorch_y = dtorch_x[dtorch_idx1, dtorch_idx2]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 3: two tensor indices with remaining dims
        torch_y = torch_x[idx1, idx2, :]
        dtorch_y = dtorch_x[dtorch_idx1, dtorch_idx2, :]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 4: broadcast tensor indices (different shapes that broadcast)
        idx_broadcast_1 = torch.tensor([0, 1, 2, 3], dtype=torch.int64).reshape(4, 1)
        idx_broadcast_2 = torch.tensor([0, 1, 2], dtype=torch.int64).reshape(1, 3)
        dtorch_idx_b1 = dtorch.Tensor(idx_broadcast_1)
        dtorch_idx_b2 = dtorch.Tensor(idx_broadcast_2)
        torch_y = torch_x[idx_broadcast_1, idx_broadcast_2]
        dtorch_y = dtorch_x[dtorch_idx_b1, dtorch_idx_b2]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 5: tensor index with scalar (integer) index mixed
        torch_y = torch_x[idx1, 3]
        dtorch_y = dtorch_x[dtorch_idx1, 3]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 6: tensor index with slice
        torch_y = torch_x[idx1, 2:5]
        dtorch_y = dtorch_x[dtorch_idx1, 2:5]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 7: test with arange (simulating the original use case)
        torch_arange = torch.arange(x_shape[0], dtype=torch.int64)
        torch_argmax = torch.randint(0, x_shape[1], (x_shape[0],), dtype=torch.int64)
        dtorch_arange = dtorch.Tensor(torch_arange)
        dtorch_argmax = dtorch.Tensor(torch_argmax)
        torch_y = torch_x[torch_arange, torch_argmax]
        dtorch_y = dtorch_x[dtorch_arange, dtorch_argmax]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

    def test_tensor_list_indexing(test_case):
        """Test advanced indexing with Python lists (aligns with PyTorch behavior)."""
        x_shape = [8, 7, 4, 6, 10]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        # Case 1: single list index (first dim)
        idx_list = [0, 1, 2]
        torch_y = torch_x[idx_list]
        dtorch_y = dtorch_x[idx_list]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 2: two list indices
        idx_list1 = [0, 1, 2]
        idx_list2 = [1, 3, 5]
        torch_y = torch_x[idx_list1, idx_list2]
        dtorch_y = dtorch_x[idx_list1, idx_list2]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 3: list index with slice (simulates the pooled_output use case)
        torch_y = torch_x[:, [1, 3, 5]]
        dtorch_y = dtorch_x[:, [1, 3, 5]]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 4: list index with scalar
        torch_y = torch_x[[0, 1, 2], 3]
        dtorch_y = dtorch_x[[0, 1, 2], 3]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 5: list index of length 1 (simulating single batch)
        torch_y = torch_x[[0], [5]]
        dtorch_y = dtorch_x[[0], [5]]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 6: arange as list with list of argmax indices
        torch_arange_list = list(range(x_shape[0]))
        torch_argmax_list = torch.randint(0, x_shape[1], (x_shape[0],)).tolist()
        torch_y = torch_x[torch_arange_list, torch_argmax_list]
        dtorch_y = dtorch_x[torch_arange_list, torch_argmax_list]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

    def test_tensor_tuple_indexing(test_case):
        """Test advanced indexing with Python tuples (aligns with PyTorch behavior)."""
        x_shape = [8, 7, 4, 6, 10]
        torch_x = torch.rand(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.Tensor(torch_x)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        # Case 1: single tuple index (first dim)
        idx_tuple = (0, 1, 2)
        torch_y = torch_x[idx_tuple]
        dtorch_y = dtorch_x[idx_tuple]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 2: two tuple indices
        idx_tuple1 = (0, 1, 2)
        idx_tuple2 = (1, 3, 5)
        torch_y = torch_x[idx_tuple1, idx_tuple2]
        dtorch_y = dtorch_x[idx_tuple1, idx_tuple2]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 3: tuple index with slice
        torch_y = torch_x[:, (1, 3, 5)]
        dtorch_y = dtorch_x[:, (1, 3, 5)]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 4: tuple index with scalar
        torch_y = torch_x[(0, 1, 2), 3]
        dtorch_y = dtorch_x[(0, 1, 2), 3]
        assert_tensor_equal(test_case, torch_y, dtorch_y)

        # Case 5: list index with tuple index mixed
        torch_y = torch_x[[0, 1, 2], (3, 4, 5)]
        dtorch_y = dtorch_x[[0, 1, 2], (3, 4, 5)]
        assert_tensor_equal(test_case, torch_y, dtorch_y)


if __name__ == "__main__":
    unittest.main()
