"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM as TransformersLlamaForCausalLM

import dtorch
from dtorch.test.test_util import (
    assert_tensor_allclose,
    check_pytorch_version,
    gen_arg_list,
)
from dtorch.test.modules.llama import LlamaForCausalLM
from dtorch.distributed_spec import init_device_mesh, Replicate


@torch.inference_mode
def _test_llama(test_case, device):
    config = LlamaConfig()
    config.num_hidden_layers = 1

    torch_in = torch.randint(low=1000, high=2000, size=(2, 16), device=device)
    torch_llama = TransformersLlamaForCausalLM(config)
    torch_llama.to(device=device)
    torch_out = torch_llama(torch_in)[0]

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_llama = LlamaForCausalLM(config)
    dtorch_llama.to(device=device)
    dtorch_llama.model.rotary_emb.to(device=device)
    dtorch_llama.load_state_dict(torch_llama.state_dict())
    dtorch_out = dtorch_llama(dtorch_in)
    assert_tensor_allclose(test_case, torch_in, dtorch_in)
    assert_tensor_allclose(test_case, torch_llama.lm_head.weight, dtorch_llama.lm_head.weight)
    assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)

    # Run with dtensor
    device_mesh = init_device_mesh(device, 2, mesh_dim_names=["dp"])
    if not dtorch.default_graph.satisfy(device_mesh):
        return
    dtorch_llama = LlamaForCausalLM(config, device_mesh=device_mesh)
    dtorch_llama.model.rotary_emb.to(device=device)
    dtorch_llama.load_state_dict(torch_llama.state_dict())
    dtorch_out = dtorch_llama(dtorch_in)
    assert_tensor_allclose(test_case, torch_llama.lm_head.weight, dtorch_llama.lm_head.weight)
    assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)

    device_mesh = init_device_mesh(device, 2, mesh_dim_names=["tp"])
    dtorch_llama = LlamaForCausalLM(config, device_mesh=device_mesh)
    dtorch_llama.model.rotary_emb.to(device=device)
    dtorch_llama.load_state_dict(torch_llama.state_dict())
    dtorch_out = dtorch_llama(dtorch_in)
    assert_tensor_allclose(test_case, torch_llama.lm_head.weight, dtorch_llama.lm_head.weight)
    assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)


class TestLlama(unittest.TestCase):
    def test_llama(test_case):
        if not check_pytorch_version(min_version="2.5.0"):
            return

        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_llama(test_case, *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


if __name__ == "__main__":
    unittest.main()
