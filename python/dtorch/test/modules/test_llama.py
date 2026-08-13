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
    is_graph_satisfy,
)
from dtorch.test.modules.llama import LlamaForCausalLM
from dtorch.distributed_spec import init_device_mesh


@torch.inference_mode
def _test_llama(test_case, device):
    # small size for testing
    config = LlamaConfig()
    config.num_hidden_layers = 2
    config.hidden_size = 1024
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.intermediate_size = 2752

    torch_in = torch.randint(low=1000, high=2000, size=(2, 16), device=device)
    torch_llama = TransformersLlamaForCausalLM(config)
    torch_llama.to(device=device)
    torch_out = torch_llama(torch_in)[0]

    def dtorch_imp(dp=1, tp=1, pp=1, ulysess_cp=1, ring_cp=1):
        device_mesh = init_device_mesh(
            device,
            (dp, tp, pp, ulysess_cp, ring_cp),
            mesh_dim_names=["dp", "tp", "pp", "ulysess_cp", "ring_cp"],
        )
        if not is_graph_satisfy(dtorch.default_graph, device_mesh):
            return

        dtorch_in = dtorch.Tensor(torch_in)
        dtorch_llama = LlamaForCausalLM(config, device_mesh=device_mesh)
        dtorch_llama.load_state_dict(torch_llama.state_dict())
        dtorch_out = dtorch_llama(dtorch_in)
        assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)
        del dtorch_llama
        dtorch.default_graph.empty_cache()

    dtorch_imp()
    dtorch_imp(dp=2)
    dtorch_imp(tp=2)
    dtorch_imp(pp=2)
    dtorch_imp(ulysess_cp=2)
    dtorch_imp(tp=2, pp=2, ulysess_cp=2)
    dtorch_imp(dp=2, tp=2, pp=2, ulysess_cp=2)


class TestLlama(unittest.TestCase):
    def test_llama(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_llama(test_case, *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


if __name__ == "__main__":
    unittest.main()
