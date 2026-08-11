"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
import os
import gc
from collections import OrderedDict

import torch
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel as TransformersFluxTransformer2DModel,
)

import dtorch
from dtorch.util.benchmark import Benchmark
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal, get_test_data_path
from dtorch.distributed_spec import init_device_mesh
from dtorch.test.system_overhead.util import print_benchmark_result
from dtorch.applications.huggingface.diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel as DTorchFluxTransformer2DModel,
)
from dtorch.cuda.nvtx import module_register_nvtx


def _benchmark_flux_transformer(test_case, device):
    warmup = 1
    count = 3
    dtype = torch.bfloat16

    transformer_model_path = os.path.join(get_test_data_path(), "FLUX.1-dev", "transformer")

    torch_hidden_states = torch.randn(1, 4096, 64, device=device, dtype=dtype)
    torch_encoder_hidden_states = torch.randn(1, 512, 4096, device=device, dtype=dtype)
    torch_pooled_projections = torch.randn(1, 768, device=device, dtype=dtype)
    torch_timestep = torch.tensor((1000.0,), device=device, dtype=torch.bfloat16)
    torch_img_ids = torch.randn(4096, 3, device=device, dtype=dtype)
    torch_txt_ids = torch.randn(512, 3, device=device, dtype=dtype)
    torch_guidance = torch.tensor((4.0,), device=device, dtype=torch.float32)

    # torch
    torch_model = TransformersFluxTransformer2DModel.from_pretrained(transformer_model_path, torch_type=dtype).eval()
    torch_model = torch_model.to(device=device, dtype=dtype)
    # module_register_nvtx(torch_model, tag="torch_model")

    @torch.inference_mode()
    def torch_func():
        for i in range(10):
            result = torch_model(
                torch_hidden_states,
                torch_encoder_hidden_states,
                torch_pooled_projections,
                torch_timestep,
                img_ids=torch_img_ids,
                txt_ids=torch_txt_ids,
                guidance=torch_guidance,
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

    ug_hidden_states = dtorch.Tensor(torch_hidden_states)
    ug_encoder_hidden_states = dtorch.Tensor(torch_encoder_hidden_states)
    ug_pooled_projections = dtorch.Tensor(torch_pooled_projections)
    ug_timestep = dtorch.Tensor(torch_timestep)
    ug_img_ids = dtorch.Tensor(torch_img_ids)
    ug_txt_ids = dtorch.Tensor(torch_txt_ids)
    ug_guidance = dtorch.Tensor(torch_guidance)

    ug_model = DTorchFluxTransformer2DModel.from_pretrained(
        transformer_model_path,
        torch_dtype=dtype,
        device_mesh=device_mesh,
    )
    # module_register_nvtx(ug_model, tag="ug_model")
    def ug_func():
        for i in range(10):
            result = ug_model(
                ug_hidden_states,
                ug_encoder_hidden_states,
                ug_pooled_projections,
                ug_timestep,
                img_ids=ug_img_ids,
                txt_ids=ug_txt_ids,
                guidance=ug_guidance,
            )[0]
        result.to_torch()
        return result

    ug_metric, ug_out = Benchmark.run(ug_func, warmup=warmup, count=count)
    assert_tensor_equal(test_case, torch_out, ug_out)

    del ug_model
    dtorch.default_graph.empty_cache()

    print_benchmark_result(
        "flux_transformer",
        device,
        warmup,
        count,
        [(list(torch_hidden_states.shape), torch_metric.duration, ug_metric.duration)],
    )


class TestFluxTransformer(unittest.TestCase):
    def test_benchmark_flux_transformer(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        for arg in gen_arg_list(arg_dict):
            _benchmark_flux_transformer(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
