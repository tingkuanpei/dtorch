"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_apply_rotary_emb(test_case, device):
    shape = (1, 8, 256, 128)
    torch_in = torch.rand(*shape, dtype=torch.float16, device=device)
    torch_cos = torch.rand(*shape[2:], dtype=torch.float32, device=device)
    torch_sin = torch.rand(*shape[2:], dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_cos = dtorch.Tensor(torch_cos)
    dtorch_sin = dtorch.Tensor(torch_sin)

    torch_out = dtorch.compiled_op.apply_rotary_emb(torch_in, torch_cos, torch_sin)
    dtorch_out = dtorch.nn.functional._apply_rotary_emb(dtorch_in, [dtorch_cos, dtorch_sin])
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_silu_linear_chunk(test_case, device):
    chunk_size = 6
    embedding_dim = 3072
    torch_in = torch.rand((1, embedding_dim), dtype=torch.float16, device=device)
    torch_weight = torch.rand((chunk_size * embedding_dim, embedding_dim), dtype=torch.float16, device=device)
    torch_bias = torch.rand((chunk_size * embedding_dim), dtype=torch.float16, device=device)
    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_weight = dtorch.Tensor(torch_weight)
    dtorch_bias = dtorch.Tensor(torch_bias)

    torch_out = dtorch.compiled_op.silu_linear_chunk(torch_in, torch_weight, torch_bias, chunk_size)
    dtorch_out = dtorch.nn.functional._silu_linear_chunk(dtorch_in, dtorch_weight, dtorch_bias, chunk_size)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_layer_norm_mul_add(test_case, device):
    torch_in = torch.rand((1, 4096, 3072), dtype=torch.float16, device=device)
    torch_scale = torch.rand((1, 3072), dtype=torch.float16, device=device)
    torch_shift = torch.rand((1, 3072), dtype=torch.float16, device=device)

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_scale = dtorch.Tensor(torch_scale)
    dtorch_shift = dtorch.Tensor(torch_shift)

    torch_out = dtorch.compiled_op.layer_norm_mul_add(torch_in, torch_scale, torch_shift, (3072,), 1e-06, None, None)
    dtorch_out = dtorch.nn.functional._layer_norm_mul_add(
        dtorch_in, dtorch_scale, dtorch_shift, (3072,), 1e-06, None, None
    )
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestFusedCompileKernel(unittest.TestCase):
    def test_apply_rotary_emb(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_apply_rotary_emb(test_case, *arg)
            _test_silu_linear_chunk(test_case, *arg)
            _test_layer_norm_mul_add(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
