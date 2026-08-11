"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
import os
import gc
from collections import OrderedDict

import torch
from diffusers import StableDiffusion3Pipeline as DiffusersStableDiffusion3Pipeline

import dtorch
from dtorch.util.benchmark import Benchmark
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal, get_test_data_path
from dtorch.distributed_spec import init_device_mesh
from dtorch.test.system_overhead.util import print_benchmark_result
from dtorch.applications.huggingface.diffusers import (
    StableDiffusion3Pipeline as DTorchStableDiffusion3Pipeline,
)
from dtorch.cuda.nvtx import module_register_nvtx


def _benchmark_text_encoder(test_case, device):
    warmup = 3
    count = 10
    dtype = torch.bfloat16

    model_path = os.path.join(get_test_data_path(), "stable-diffusion-3-medium-diffusers")

    torch_input_ids_1 = torch.randint(1000, 2000, (1, 77), device=device)
    torch_input_ids_2 = torch.randint(1000, 2000, (1, 77), device=device)
    torch_input_ids_3 = torch.randint(1000, 2000, (1, 256), device=device)
    pooled_output_index_1 = torch_input_ids_1.argmax(dim=-1).tolist()
    pooled_output_index_2 = torch_input_ids_2.argmax(dim=-1).tolist()

    # torch
    torch_pipe = DiffusersStableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype)
    torch_pipe = torch_pipe.to(device=device)
    # module_register_nvtx(torch_pipe.text_encoder, "text_encoder")
    # module_register_nvtx(torch_pipe.text_encoder_2, "text_encoder_2")
    # module_register_nvtx(torch_pipe.text_encoder_3, "text_encoder_3")

    @torch.inference_mode()
    def torch_func():
        out_1 = torch_pipe.text_encoder(torch_input_ids_1, output_hidden_states=True).last_hidden_state
        out_2 = torch_pipe.text_encoder_2(torch_input_ids_2, output_hidden_states=True).last_hidden_state
        out_3 = torch_pipe.text_encoder_3(torch_input_ids_3)[0]
        out_1 = torch_pipe.text_encoder(torch_input_ids_1, output_hidden_states=True).last_hidden_state
        out_2 = torch_pipe.text_encoder_2(torch_input_ids_2, output_hidden_states=True).last_hidden_state
        out_3 = torch_pipe.text_encoder_3(torch_input_ids_3)[0]
        return out_1, out_2, out_3

    torch_metric, (torch_out_1, torch_out_2, torch_out_3) = Benchmark.run(torch_func, warmup=warmup, count=count)

    # Clean up torch model to free GPU memory before loading DTorch model
    del torch_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # dtorch
    device_mesh = init_device_mesh(
        device,
        (1, 1),
        mesh_dim_names=["dp", "tp"],
    )

    dtorch_input_ids_1 = dtorch.Tensor(torch_input_ids_1)
    dtorch_input_ids_2 = dtorch.Tensor(torch_input_ids_2)
    dtorch_input_ids_3 = dtorch.Tensor(torch_input_ids_3)

    dtorch_pipe = DTorchStableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_mesh=device_mesh,
    )
    # module_register_nvtx(dtorch_pipe.text_encoder, "text_encoder")
    # module_register_nvtx(dtorch_pipe.text_encoder_2, "text_encoder_2")
    # module_register_nvtx(dtorch_pipe.text_encoder_3, "text_encoder_3")

    def dtorch_func():
        out_1 = dtorch_pipe.text_encoder(
            dtorch_input_ids_1, output_hidden_states=True, pooled_output_index=pooled_output_index_1
        ).last_hidden_state
        out_2 = dtorch_pipe.text_encoder_2(
            dtorch_input_ids_2, output_hidden_states=True, pooled_output_index=pooled_output_index_2
        ).last_hidden_state
        out_3 = dtorch_pipe.text_encoder_3(dtorch_input_ids_3)[0]
        out_1 = dtorch_pipe.text_encoder(
            dtorch_input_ids_1, output_hidden_states=True, pooled_output_index=pooled_output_index_1
        ).last_hidden_state
        out_2 = dtorch_pipe.text_encoder_2(
            dtorch_input_ids_2, output_hidden_states=True, pooled_output_index=pooled_output_index_2
        ).last_hidden_state
        out_3 = dtorch_pipe.text_encoder_3(dtorch_input_ids_3)[0]
        return out_1, out_2, out_3

    dtorch_metric, (dtorch_out_1, dtorch_out_2, dtorch_out_3) = Benchmark.run(dtorch_func, warmup=warmup, count=count)
    assert_tensor_equal(test_case, torch_out_1, dtorch_out_1)
    assert_tensor_equal(test_case, torch_out_2, dtorch_out_2)
    assert_tensor_equal(test_case, torch_out_3, dtorch_out_3)

    del dtorch_pipe
    dtorch.default_graph.empty_cache()

    shape = [list(torch_input_ids_1.shape), list(torch_input_ids_2.shape), list(torch_input_ids_3.shape)]
    print_benchmark_result(
        "text_encoders",
        device,
        warmup,
        count,
        [(shape, torch_metric.duration, dtorch_metric.duration)],
    )


class TestClip(unittest.TestCase):
    def test_benchmark_clip(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        for arg in gen_arg_list(arg_dict):
            _benchmark_text_encoder(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
