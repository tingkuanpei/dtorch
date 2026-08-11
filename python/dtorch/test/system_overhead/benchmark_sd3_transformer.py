"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
import os
import gc
from collections import OrderedDict

import torch
from diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel as TransformersSD3Transformer2DModel,
)

import dtorch
from dtorch.util.benchmark import Benchmark
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal, get_test_data_path
from dtorch.distributed_spec import init_device_mesh
from dtorch.test.system_overhead.util import print_benchmark_result
from dtorch.applications.huggingface.diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel as DTorchSD3Transformer2DModel,
)


def _benchmark_sd3_transformer(test_case, device):
    warmup = 1
    count = 3
    dtype = torch.bfloat16

    transformer_model_path = os.path.join(get_test_data_path(), "stable-diffusion-3-medium-diffusers", "transformer")

    torch_hidden_states = torch.randn(2, 16, 128, 128, device=device, dtype=dtype)
    torch_encoder_hidden_states = torch.randn(2, 333, 4096, device=device, dtype=dtype)
    torch_pooled_projections = torch.randn(2, 2048, device=device, dtype=dtype)
    torch_timestep = torch.tensor((1000.0, 1000.0), device=device, dtype=torch.float32)

    # torch
    torch_model = TransformersSD3Transformer2DModel.from_pretrained(transformer_model_path, torch_type=dtype).eval()
    torch_model = torch_model.to(device=device, dtype=dtype)

    @torch.inference_mode()
    def torch_func():
        for i in range(28):
            result = torch_model(
                torch_hidden_states,
                torch_encoder_hidden_states,
                torch_pooled_projections,
                torch_timestep,
            )
            result = result[0]
        return result

    torch_metric, torch_out = Benchmark.run(torch_func, warmup=warmup, count=count)

    del torch_model
    gc.collect()
    torch.cuda.empty_cache()

    # dtorch
    device_mesh = init_device_mesh(
        device,
        (1, 1, 1, 1),
        mesh_dim_names=["dp", "tp", "ulysess_cp", "ring_cp"],
    )

    dtorch_hidden_states = dtorch.Tensor(torch_hidden_states)
    dtorch_encoder_hidden_states = dtorch.Tensor(torch_encoder_hidden_states)
    dtorch_pooled_projections = dtorch.Tensor(torch_pooled_projections)
    dtorch_timestep = dtorch.Tensor(torch_timestep)

    dtorch_model = DTorchSD3Transformer2DModel.from_pretrained(
        transformer_model_path,
        torch_dtype=dtype,
        device_mesh=device_mesh,
    )

    def dtorch_func():
        for i in range(28):
            result = dtorch_model(
                dtorch_hidden_states,
                dtorch_encoder_hidden_states,
                dtorch_pooled_projections,
                dtorch_timestep,
            )[0]
        result.to_torch()
        return result

    dtorch_metric, dtorch_out = Benchmark.run(dtorch_func, warmup=warmup, count=count)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    del dtorch_model
    dtorch.default_graph.empty_cache()

    print_benchmark_result(
        "sd3_transformer",
        device,
        warmup,
        count,
        [(list(torch_hidden_states.shape), torch_metric.duration, dtorch_metric.duration)],
    )


class TestSD3Transformer(unittest.TestCase):
    def test_benchmark_sd3_transformer(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        for arg in gen_arg_list(arg_dict):
            _benchmark_sd3_transformer(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
