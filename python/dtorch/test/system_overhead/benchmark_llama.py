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
from dtorch.util.benchmark import Benchmark
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose
from dtorch.test.modules.llama import LlamaForCausalLM


def _benchmark_llama(test_case, device):
    warmup = 5
    count = 30

    config = LlamaConfig()
    config.num_hidden_layers = 10

    torch_in = torch.randint(low=1000, high=2000, size=(2, 1024), device=device)

    # torch
    torch_llama = TransformersLlamaForCausalLM(config)
    torch_llama.to(device=device)

    @torch.inference_mode()
    def torch_func():
        return torch_llama(torch_in)[0]

    torch_metric, torch_out = Benchmark.run(torch_func, warmup=warmup, count=count)

    # dtorch
    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_llama = LlamaForCausalLM(config)
    dtorch_llama.to(device=device)
    dtorch_llama.model.rotary_emb.to(device=device)
    dtorch_llama.load_state_dict(torch_llama.state_dict())

    def dtorch_func():
        return dtorch_llama(dtorch_in)

    dtorch_metric, dtorch_out = Benchmark.run(dtorch_func, warmup=warmup, count=count)
    assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)

    print(f"{dtorch_metric.duration=}. {torch_metric.duration=}")


class TestLlama(unittest.TestCase):
    def test_benchmark_llama(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        for arg in gen_arg_list(arg_dict):
            _benchmark_llama(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
