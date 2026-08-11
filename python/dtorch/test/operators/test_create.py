"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest

import torch

import dtorch
from dtorch.test.test_util import assert_tensor_equal


class TestCreate(unittest.TestCase):
    def test_tensor_create_api(test_case):
        x_shape = [8, 7, 4, 6]

        torch_x = torch.zeros(x_shape, dtype=torch.float32)
        dtorch_x = dtorch.zeros(x_shape, dtype=torch.float32)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.zeros(*x_shape, dtype=torch.float32)
        dtorch_x = dtorch.zeros(*x_shape, dtype=torch.float32)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        dtorch_x = dtorch.from_torch(torch_x)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.zeros_like(torch_x)
        dtorch_x = dtorch.zeros_like(dtorch_x)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.zeros(x_shape, dtype=torch.float32, device="cpu")
        dtorch_x = dtorch.zeros(x_shape, dtype=torch.float32, device="cpu")
        assert_tensor_equal(test_case, torch_x, dtorch_x)
        test_case.assertTrue(dtorch_x.device == torch_x.device)

        torch_x = torch.ones(x_shape, dtype=torch.float32)
        dtorch_x = dtorch.ones(x_shape, dtype=torch.float32)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.full(x_shape, 1.0, dtype=torch.float32)
        dtorch_x = dtorch.full(x_shape, 1.0, dtype=torch.float32)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.full_like(torch_x, 2.0)
        dtorch_x = dtorch.full_like(dtorch_x, 2.0)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.arange(1.0, 6.01, 0.6, dtype=torch.float32)
        dtorch_x = dtorch.arange(1.0, 6.01, 0.6, dtype=torch.float32)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.arange(1.0, 6.01, 0.6)
        dtorch_x = dtorch.arange(1.0, 6.01, 0.6)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

    def test_random_tensor_create_api(test_case):
        torch_generator = torch.Generator()
        torch_generator.manual_seed(42)
        dtorch_generator = torch.Generator()
        dtorch_generator.manual_seed(42)

        torch_x = torch.randint(3, 10, (2, 2), generator=torch_generator)
        dtorch_x = dtorch.randint(3, 10, (2, 2), generator=dtorch_generator)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.rand(2, 3, generator=torch_generator)
        dtorch_x = dtorch.rand(2, 3, generator=dtorch_generator)
        assert_tensor_equal(test_case, torch_x, dtorch_x)

        torch_x = torch.randn(2, 3, generator=torch_generator)
        dtorch_x = dtorch.randn(2, 3, generator=dtorch_generator)
        assert_tensor_equal(test_case, torch_x, dtorch_x)


if __name__ == "__main__":
    unittest.main()
