"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal
import dtorch.nn.functional as F


def _test_batch_norm(test_case, shape, device):
    num_features = shape[1]
    eps = 1e-5
    momentum = 0.1
    training = False

    torch_input = torch.rand(*shape, dtype=torch.float32, device=device)
    torch_mean = torch.rand(num_features, dtype=torch.float32, device=device)
    torch_var = torch.rand(num_features, dtype=torch.float32, device=device)
    torch_scale = torch.rand(num_features, dtype=torch.float32, device=device)
    torch_bias = torch.rand(num_features, dtype=torch.float32, device=device)

    dtorch_input = dtorch.Tensor(torch_input)
    dtorch_mean = dtorch.Tensor(torch_mean)
    dtorch_var = dtorch.Tensor(torch_var)
    dtorch_scale = dtorch.Tensor(torch_scale)
    dtorch_bias = dtorch.Tensor(torch_bias)

    torch_out = torch.nn.functional.batch_norm(
        torch_input,
        torch_mean,
        torch_var,
        torch_scale,
        torch_bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )

    dtorch_out = dtorch.nn.functional.batch_norm(
        dtorch_input,
        dtorch_mean,
        dtorch_var,
        dtorch_scale,
        dtorch_bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )

    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.batch_norm(
        torch_input,
        torch_mean,
        torch_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )

    dtorch_out = dtorch.nn.functional.batch_norm(
        dtorch_input,
        dtorch_mean,
        dtorch_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )

    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestBatchNorm(unittest.TestCase):
    def test_batch_norm(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 16, 8, 8)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_batch_norm(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
