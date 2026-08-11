"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict
from typing import Optional

import torch

import dtorch
from dtorch.distributed_spec import init_device_mesh, Shard
from dtorch.test.test_util import (
    gen_arg_list,
    assert_tensor_allclose,
    assert_tensor_equal,
    check_pytorch_version,
)


def _test_scaled_dot_product_attention(test_case, device):
    torch_query = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_key = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_value = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_out = torch.nn.functional.scaled_dot_product_attention(torch_query, torch_key, torch_value, enable_gqa=True)

    dtorch_query = dtorch.Tensor(torch_query)
    dtorch_key = dtorch.Tensor(torch_key)
    dtorch_value = dtorch.Tensor(torch_value)
    dtorch_out = dtorch.nn.functional.scaled_dot_product_attention(
        dtorch_query, dtorch_key, dtorch_value, enable_gqa=True
    )

    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_scaled_dot_product_attention_with_cp(test_case, device):
    torch_query = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_key = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_value = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_out = torch.nn.functional.scaled_dot_product_attention(torch_query, torch_key, torch_value)

    dtorch_query = dtorch.Tensor(torch_query)
    dtorch_key = dtorch.Tensor(torch_key)
    dtorch_value = dtorch.Tensor(torch_value)
    dtorch_out = dtorch.nn.functional.scaled_dot_product_attention(dtorch_query, dtorch_key, dtorch_value)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    dp = 2
    ulysess_cp = 2
    ring_cp = 2
    device_mesh = init_device_mesh(
        device,
        (dp, ulysess_cp, ring_cp),
        mesh_dim_names=["dp", "ulysess_cp", "ring_cp"],
    )
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    placements = [Shard(0), Shard(2), Shard(2)]
    dtorch_query = dtorch.Tensor(torch_query, device_mesh=device_mesh, placements=placements)
    dtorch_key = dtorch.Tensor(torch_key, device_mesh=device_mesh, placements=placements)
    dtorch_value = dtorch.Tensor(torch_value, device_mesh=device_mesh, placements=placements)
    dtorch_out = dtorch.nn.functional.scaled_dot_product_attention(
        dtorch_query,
        dtorch_key,
        dtorch_value,
    )
    assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-3, atol=1e-3)


def _test_joint_scaled_dot_product_attention_with_cp(test_case, device):
    # torch
    torch_query = torch.rand(8, 32, 512, 64, dtype=torch.float16, device=device)
    torch_key = torch.rand(8, 32, 512, 64, dtype=torch.float16, device=device)
    torch_value = torch.rand(8, 32, 512, 64, dtype=torch.float16, device=device)
    torch_other_query = torch.rand(8, 32, 128, 64, dtype=torch.float16, device=device)
    torch_other_key = torch.rand(8, 32, 128, 64, dtype=torch.float16, device=device)
    torch_other_value = torch.rand(8, 32, 128, 64, dtype=torch.float16, device=device)

    torch_query_concat = torch.cat([torch_query, torch_other_query], dim=2)
    torch_key_concat = torch.cat([torch_key, torch_other_key], dim=2)
    torch_value_concat = torch.cat([torch_value, torch_other_value], dim=2)
    torch_out_concat = torch.nn.functional.scaled_dot_product_attention(
        torch_query_concat,
        torch_key_concat,
        torch_value_concat,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    )
    length_query = torch_query.shape[2]
    torch_out, torch_other_out = (
        torch_out_concat[:, :, :length_query],
        torch_out_concat[:, :, length_query:],
    )

    # dtorch local device
    dtorch_query = dtorch.Tensor(torch_query)
    dtorch_key = dtorch.Tensor(torch_key)
    dtorch_value = dtorch.Tensor(torch_value)
    dtorch_other_query = dtorch.Tensor(torch_other_query)
    dtorch_other_key = dtorch.Tensor(torch_other_key)
    dtorch_other_value = dtorch.Tensor(torch_other_value)

    dtorch_query_concat = dtorch.cat([dtorch_query, dtorch_other_query], dim=2)
    dtorch_key_concat = dtorch.cat([dtorch_key, dtorch_other_key], dim=2)
    dtorch_value_concat = dtorch.cat([dtorch_value, dtorch_other_value], dim=2)
    dtorch_out_concat = dtorch.nn.functional.scaled_dot_product_attention(
        dtorch_query_concat,
        dtorch_key_concat,
        dtorch_value_concat,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    )
    length_query = dtorch_query.shape[2]
    dtorch_out, dtorch_other_out = (
        dtorch_out_concat[:, :, :length_query],
        dtorch_out_concat[:, :, length_query:],
    )
    assert_tensor_equal(test_case, torch_out, dtorch_out)
    assert_tensor_equal(test_case, torch_other_out, dtorch_other_out)

    # dtorch distributed device
    dp = 2
    ulysess_cp = 2
    ring_cp = 2
    device_mesh = init_device_mesh(
        device,
        (dp, ulysess_cp, ring_cp),
        mesh_dim_names=["dp", "ulysess_cp", "ring_cp"],
    )
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    placements = [Shard(0), Shard(2), Shard(2)]
    dtorch_query = dtorch.Tensor(torch_query, device_mesh=device_mesh, placements=placements)
    dtorch_key = dtorch.Tensor(torch_key, device_mesh=device_mesh, placements=placements)
    dtorch_value = dtorch.Tensor(torch_value, device_mesh=device_mesh, placements=placements)
    dtorch_other_query = dtorch.Tensor(torch_other_query, device_mesh=device_mesh, placements=placements)
    dtorch_other_key = dtorch.Tensor(torch_other_key, device_mesh=device_mesh, placements=placements)
    dtorch_other_value = dtorch.Tensor(torch_other_value, device_mesh=device_mesh, placements=placements)
    dtorch_query_concat = dtorch.cat([dtorch_query, dtorch_other_query], dim=2)
    dtorch_key_concat = dtorch.cat([dtorch_key, dtorch_other_key], dim=2)
    dtorch_value_concat = dtorch.cat([dtorch_value, dtorch_other_value], dim=2)

    dtorch_out_concat = dtorch.nn.functional.scaled_dot_product_attention(
        dtorch_query_concat,
        dtorch_key_concat,
        dtorch_value_concat,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    )
    length_query = dtorch_query.shape[2]
    dtorch_out, dtorch_other_out = (
        dtorch_out_concat[:, :, :length_query],
        dtorch_out_concat[:, :, length_query:],
    )
    assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-3, atol=1e-3)
    assert_tensor_allclose(test_case, torch_other_out, dtorch_other_out, rtol=1e-3, atol=1e-3)


def _test_sage_attention(test_case, is_causal):
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
    except ImportError:
        return

    device = "cuda"
    torch_query = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_key = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_value = torch.rand(32, 32, 128, 64, dtype=torch.float16, device=device)
    torch_out = torch.nn.functional.scaled_dot_product_attention(torch_query, torch_key, torch_value, enable_gqa=True)
    sage_attn_out = sageattn_qk_int8_pv_fp16_cuda(
        torch_query,
        torch_key,
        torch_value,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran="per_thread",
    )

    dtorch_query = dtorch.Tensor(torch_query)
    dtorch_key = dtorch.Tensor(torch_key)
    dtorch_value = dtorch.Tensor(torch_value)
    dtorch_out = dtorch.nn.functional.scaled_dot_product_attention(
        dtorch_query,
        dtorch_key,
        dtorch_value,
        is_causal=is_causal,
        sdpa_option=dtorch.SdpaOption(sage_attn_type="qk_int8_pv_fp16"),
    )

    assert_tensor_equal(test_case, sage_attn_out, dtorch_out)

    # # dtorch distributed device
    # dp = 2
    # ulysess_cp = 2
    # ring_cp = 2
    # device_mesh = init_device_mesh(device, (dp, ulysess_cp, ring_cp), mesh_dim_names=["dp", "ulysess_cp", "ring_cp"])
    # if not dtorch.default_graph.satisfy(device_mesh):
    #     return
    # placements = [Shard(0), Shard(2), Shard(2)]
    # dtorch_query = dtorch.Tensor(
    #     torch_query, device_mesh=device_mesh, placements=placements
    # )
    # dtorch_key = dtorch.Tensor(torch_key, device_mesh=device_mesh, placements=placements)
    # dtorch_value = dtorch.Tensor(
    #     torch_value, device_mesh=device_mesh, placements=placements
    # )

    # dtorch_out = dtorch.nn.functional.scaled_dot_product_attention(
    #     dtorch_query,
    #     dtorch_key,
    #     dtorch_value,
    #     is_causal=is_causal,
    #     sdpa_option=dtorch.SdpaOption(
    #         dp=dp,
    #         ulysess_cp=ulysess_cp,
    #         ring_cp=ring_cp,
    #         sage_attn_type="auto",
    #     ),
    # )
    # assert_tensor_allclose(test_case, sage_attn_out, dtorch_out, rtol=1e-3, atol=1e-3)


class TestScaledDotProductAttention(unittest.TestCase):
    def test_scaled_dot_product_attention(test_case):
        if not check_pytorch_version(min_version="2.5.0"):
            return

        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_scaled_dot_product_attention(test_case, *arg)
            _test_scaled_dot_product_attention_with_cp(test_case, *arg)
            _test_joint_scaled_dot_product_attention_with_cp(test_case, *arg)

        # arg_dict = OrderedDict()
        # arg_dict["is_causal"] = [True, False]
        # for arg in gen_arg_list(arg_dict):
        #     _test_sage_attention


if __name__ == "__main__":
    unittest.main()
