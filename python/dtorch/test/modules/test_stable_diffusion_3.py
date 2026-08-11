"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
import os
from collections import OrderedDict
import gc

import numpy as np
import torch
from transformers.models.clip.modeling_clip import (
    CLIPTextModelWithProjection as TransformersCLIPTextModelWithProjection,
)
from transformers.models.t5.modeling_t5 import (
    T5EncoderModel as TransformersT5EncoderModel,
)
from diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel as TransformersSD3Transformer2DModel,
)
from diffusers.models.autoencoders.autoencoder_kl import (
    AutoencoderKL as DiffusersAutoencoderKL,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchEulerDiscreteScheduler,
)
from diffusers import StableDiffusion3Pipeline as DiffusersStableDiffusion3Pipeline

import dtorch
from dtorch.test.test_util import (
    assert_tensor_equal,
    assert_tensor_allclose,
    assert_tensor_meanclose,
    gen_arg_list,
    get_test_data_path,
    is_test_data_exists,
    MemoryRequire,
    is_graph_satisfy,
    print_all_gpu_memory,
)
from dtorch.distributed_spec import init_device_mesh
from dtorch.applications.huggingface.diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel as DTorchSD3Transformer2DModel,
)
from dtorch.applications.huggingface.transformers.models.clip.modeling_clip import (
    CLIPTextModelWithProjection as DTorchCLIPTextModelWithProjection,
)
from dtorch.applications.huggingface.transformers.models.t5.modeling_t5 import (
    T5EncoderModel as DTorchT5EncoderModel,
)
from dtorch.applications.huggingface.diffusers.models.autoencoders.autoencoder_kl import (
    AutoencoderKL as DTorchAutoencoderKL,
)
from dtorch.applications.huggingface.diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as DTorchFlowMatchEulerDiscreteScheduler,
)
from dtorch.applications.huggingface.diffusers import (
    StableDiffusion3Pipeline as DTorchStableDiffusion3Pipeline,
)
from dtorch.applications.core.util.image_metrics import compute_psnr


def _test_clip(test_case, model_path, model_string, device, dtype):
    clip_model_path = os.path.join(model_path, model_string)
    torch_input_ids = torch.randint(1000, 2000, (2, 64), device=device)

    def torch_imp():
        torch_model = TransformersCLIPTextModelWithProjection.from_pretrained(clip_model_path, torch_dtype=dtype).eval()
        torch_model = torch_model.to(device=device)
        torch_output = torch_model(torch_input_ids, output_hidden_states=True)
        torch_output = torch_output.last_hidden_state
        del torch_model
        gc.collect()
        torch.cuda.empty_cache()
        return torch_output

    torch_output = torch_imp()

    def dtorch_imp(dp=1, tp=1):
        device_mesh = init_device_mesh(
            device,
            (dp, tp),
            mesh_dim_names=["dp", "tp"],
        )

        if not is_graph_satisfy(dtorch.default_graph, device_mesh):
            return

        dtorch_input_ids = dtorch.Tensor(torch_input_ids)

        dtorch_model = DTorchCLIPTextModelWithProjection.from_pretrained(
            clip_model_path,
            torch_dtype=dtype,
            device_mesh=device_mesh,
        )
        dtorch_output = dtorch_model(dtorch_input_ids, output_hidden_states=True)
        dtorch_output = dtorch_output.last_hidden_state

        assert_tensor_equal(test_case, torch_input_ids, dtorch_input_ids)
        if device_mesh.is_distributed:
            assert_tensor_allclose(test_case, torch_output, dtorch_output, rtol=1e-1, atol=1e-1)
        else:
            assert_tensor_equal(test_case, torch_output, dtorch_output)
        del dtorch_model
        dtorch.default_graph.empty_cache()

    dtorch_imp()
    dtorch_imp(dp=2)
    dtorch_imp(tp=2)
    dtorch_imp(dp=2, tp=2)


def _test_t5(test_case, model_path, small_fake_model, device, dtype):
    model_size = 0.6 if small_fake_model else 9
    memory_require = MemoryRequire(model_size, model_size, activation_gb=6)

    t5_model_path = os.path.join(model_path, "text_encoder_3")
    torch_input_ids = torch.randint(1000, 2000, (2, 256), device=device)
    torch_wo_weight_dtype = None

    def torch_imp():
        if not is_graph_satisfy(dtorch.default_graph, memory_require=memory_require):
            return

        # The T5EncoderModel has a "_keep_in_fp32_modules" parameter. Specifying the "torch_dtype" parameter when calling
        # "from_pretrained" ensures its effectiveness.
        torch_model = TransformersT5EncoderModel.from_pretrained(t5_model_path, torch_dtype=dtype).eval()
        nonlocal torch_wo_weight_dtype
        torch_wo_weight_dtype = torch_model.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype

        torch_model = torch_model.to(device=device)
        torch_output = torch_model(torch_input_ids)
        torch_output = torch_output[0]
        del torch_model
        gc.collect()
        torch.cuda.empty_cache()
        return torch_output

    torch_output = torch_imp()

    def dtorch_imp(dp=1, tp=1):
        device_mesh = init_device_mesh(
            device,
            (dp, tp),
            mesh_dim_names=["dp", "tp"],
        )

        if not is_graph_satisfy(dtorch.default_graph, device_mesh, memory_require):
            return

        dtorch_input_ids = dtorch.Tensor(torch_input_ids)

        dtorch_model = DTorchT5EncoderModel.from_pretrained(
            t5_model_path,
            torch_dtype=dtype,
            device_mesh=device_mesh,
        )
        dtorch_wo_weight_dtype = dtorch_model.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype
        test_case.assertTrue(torch_wo_weight_dtype == dtorch_wo_weight_dtype)

        dtorch_output = dtorch_model(dtorch_input_ids)
        dtorch_output = dtorch_output[0]

        assert_tensor_equal(test_case, torch_input_ids, dtorch_input_ids)
        if device_mesh.is_distributed:
            assert_tensor_allclose(test_case, torch_output, dtorch_output, rtol=1e-1, atol=1e-1)
        else:
            assert_tensor_equal(test_case, torch_output, dtorch_output)
        del dtorch_model
        dtorch.default_graph.empty_cache()

    dtorch_imp()
    dtorch_imp(dp=2)
    dtorch_imp(tp=2)
    dtorch_imp(dp=2, tp=2)


def _test_vae(test_case, model_path, device, dtype):
    vae_model_path = os.path.join(model_path, "vae")
    torch_model = DiffusersAutoencoderKL.from_pretrained(vae_model_path)
    torch_model = torch_model.to(device=device, dtype=dtype).eval()
    torch_input = torch.randn(1, 16, 32, 32, device=device, dtype=dtype)
    torch_out = torch_model.decode(torch_input)
    torch_out = torch_out[0]

    dtorch_model = DTorchAutoencoderKL.from_pretrained(vae_model_path)
    dtorch_model = dtorch_model.to(device=device, dtype=dtype)
    dtorch_input = dtorch.Tensor(torch_input)
    dtorch_out = dtorch_model.decode(dtorch_input)
    dtorch_out = dtorch_out[0]

    assert_tensor_equal(test_case, torch_input, dtorch_input)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


def _test_transformer_sd3(test_case, model_path, small_fake_model, device, dtype):
    model_size = 0.5 if small_fake_model else 4
    memory_require = MemoryRequire(model_size, model_size)

    transformer_model_path = os.path.join(model_path, "transformer")
    torch_hidden_states = torch.randn(2, 16, 128, 128, device=device, dtype=dtype)
    torch_encoder_hidden_states = torch.randn(2, 333, 4096, device=device, dtype=dtype)
    torch_pooled_projections = torch.randn(2, 2048, device=device, dtype=dtype)
    # Scale down the pooled projections to avoid overflow in the attention mechanism
    torch_pooled_projections = torch_pooled_projections / 10
    torch_timestep = torch.tensor((1000.0, 1000.0), device=device, dtype=torch.float32)

    def torch_imp():
        if not is_graph_satisfy(dtorch.default_graph, memory_require=memory_require):
            return

        torch_model = TransformersSD3Transformer2DModel.from_pretrained(
            transformer_model_path,
            torch_dtype=dtype,
        ).eval()
        torch_model = torch_model.to(device=device)
        torch_output = torch_model(
            torch_hidden_states,
            torch_encoder_hidden_states,
            torch_pooled_projections,
            torch_timestep,
        )
        torch_output = torch_output[0]
        del torch_model
        gc.collect()
        torch.cuda.empty_cache()
        return torch_output

    torch_output = torch_imp()

    def dtorch_imp(dp=1, tp=1, ulysess_cp=1, ring_cp=1):
        device_mesh = init_device_mesh(
            device,
            (dp, tp, ulysess_cp, ring_cp),
            mesh_dim_names=["dp", "tp", "ulysess_cp", "ring_cp"],
        )
        if not is_graph_satisfy(dtorch.default_graph, device_mesh, memory_require):
            return

        dtorch_hidden_states = dtorch.Tensor(torch_hidden_states)
        dtorch_encoder_hidden_states = dtorch.Tensor(torch_encoder_hidden_states)
        dtorch_pooled_projections = dtorch.Tensor(torch_pooled_projections)
        dtorch_timestep = dtorch.Tensor(torch_timestep)

        dtorch_model = DTorchSD3Transformer2DModel.from_pretrained(
            transformer_model_path,
            torch_dtype=dtype,
            device_mesh=device_mesh,
        )
        dtorch_output = dtorch_model(
            dtorch_hidden_states,
            dtorch_encoder_hidden_states,
            dtorch_pooled_projections,
            dtorch_timestep,
        )
        dtorch_output = dtorch_output[0]

        if device_mesh.is_distributed:
            assert_tensor_allclose(test_case, torch_output, dtorch_output, rtol=2e-1, atol=2e-1, equal_nan=True)
        else:
            assert_tensor_equal(test_case, torch_output, dtorch_output)
        del dtorch_model
        dtorch.default_graph.empty_cache()

    dtorch_imp()
    # dtorch_imp(dp=2)
    # dtorch_imp(tp=2)
    # dtorch_imp(ulysess_cp=2, ring_cp=2)
    # dtorch_imp(dp=2, tp=2, ulysess_cp=2, ring_cp=2)


def _test_scheduler(test_case, model_path, device, dtype):
    dtype = torch.float
    num_inference_steps = 50
    scheduler_model_path = os.path.join(model_path, "scheduler")
    torch_scheduler = DiffusersFlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_model_path)
    torch_noise_pred = torch.randn(1, 16, 128, 128, device=device, dtype=dtype)
    torch_t = torch.tensor(616.4974975585938, device=device, dtype=dtype)
    torch_latents = torch.randn(1, 16, 128, 128, device=device, dtype=dtype)
    torch_scheduler.set_timesteps(num_inference_steps, device=device)
    torch_output = torch_scheduler.step(torch_noise_pred, torch_t, torch_latents, return_dict=False)[0]

    dtorch_scheduler = DTorchFlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_model_path)
    dtorch_noise_pred = dtorch.Tensor(torch_noise_pred)
    dtorch_t = dtorch.Tensor(torch_t)
    dtorch_latents = dtorch.Tensor(torch_latents)
    dtorch_scheduler.set_timesteps(num_inference_steps, device=device)
    dtorch_output = dtorch_scheduler.step(dtorch_noise_pred, dtorch_t, dtorch_latents, return_dict=False)[0]

    assert_tensor_equal(test_case, torch_output, dtorch_output)


def _test_pipeline(test_case, model_path, small_fake_model, device, dtype):
    model_size = 1.2 if small_fake_model else 15
    same_device_plus = 0.25 if small_fake_model else 4
    memory_require = MemoryRequire(model_size, same_device_plus)
    torch_wo_weight_dtype = None

    def torch_imp():
        if not is_graph_satisfy(dtorch.default_graph, memory_require=memory_require):
            return

        torch_pipe = DiffusersStableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype)
        nonlocal torch_wo_weight_dtype
        torch_wo_weight_dtype = torch_pipe.text_encoder_3.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype
        torch_pipe = torch_pipe.to(device=device)
        torch_image = torch_pipe(
            "A cat holding a sign that says hello world",
            negative_prompt="",
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=7.0,
            generator=torch.Generator("cuda").manual_seed(1000),
        ).images[0]
        torch_array = np.asarray(torch_image)
        del torch_pipe
        gc.collect()
        torch.cuda.empty_cache()
        return torch_array

    torch_array = torch_imp()

    def dtorch_imp(dp=1, tp=1, ulysess_cp=1, ring_cp=1):
        device_mesh = init_device_mesh(
            device,
            (dp, tp, ulysess_cp, ring_cp),
            mesh_dim_names=["dp", "tp", "ulysess_cp", "ring_cp"],
        )
        if not is_graph_satisfy(dtorch.default_graph, device_mesh, memory_require):
            return

        dtorch_pipe = DTorchStableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_mesh=device_mesh,
        )
        dtorch_wo_weight_dtype = dtorch_pipe.text_encoder_3.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype
        test_case.assertTrue(torch_wo_weight_dtype, dtorch_wo_weight_dtype)

        dtorch_image = dtorch_pipe(
            "A cat holding a sign that says hello world",
            negative_prompt="",
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=7.0,
            generator=torch.Generator("cuda").manual_seed(1000),
        ).images[0]
        dtorch_array = np.asarray(dtorch_image)

        if device_mesh.is_distributed:
            if not small_fake_model:
                assert_tensor_allclose(test_case, torch_array, dtorch_array, atol=15)
                assert_tensor_meanclose(test_case, torch_array, dtorch_array, atol=1.5)
            else:
                assert_tensor_allclose(test_case, torch_array, dtorch_array, atol=4)
            test_case.assertTrue(compute_psnr(torch_array, dtorch_array) > 40)
        else:
            assert_tensor_equal(test_case, torch_array, dtorch_array)
        del dtorch_pipe
        dtorch.default_graph.empty_cache()

    dtorch_imp()
    # dtorch_imp(dp=2)
    # dtorch_imp(tp=2)
    # dtorch_imp(ulysess_cp=2, ring_cp=2)
    # dtorch_imp(dp=2, tp=2, ulysess_cp=2, ring_cp=2)


def _get_model_paths():
    """Get list of (model_path, small_fake_model) tuples for available SD3 model paths."""
    paths = []
    # small_fake_model_path = os.path.join(get_test_data_path(), "stable-diffusion-3-medium-diffusers-small-fake")
    # if os.path.exists(os.path.join(small_fake_model_path, "text_encoder", "model.safetensors")):
    #     paths.append((small_fake_model_path, True))
    model_path = os.path.join(get_test_data_path(), "stable-diffusion-3-medium-diffusers")
    if os.path.exists(os.path.join(model_path, "text_encoder", "model.safetensors")):
        paths.append((model_path, False))
    return paths


class TestClip(unittest.TestCase):
    @torch.inference_mode
    def test_clip(self):
        if not is_test_data_exists():
            return
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["dtype"] = [torch.float16]
        for arg in gen_arg_list(arg_dict):
            for model_path, _ in _get_model_paths():
                _test_clip(self, model_path, "text_encoder", *arg)
                _test_clip(self, model_path, "text_encoder_2", *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


class TestT5(unittest.TestCase):
    @torch.inference_mode
    def test_t5(self):
        if not is_test_data_exists():
            return
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["dtype"] = [torch.float16]
        for arg in gen_arg_list(arg_dict):
            for model_path, is_small_fake_model in _get_model_paths():
                _test_t5(self, model_path, is_small_fake_model, *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


class TestTransformerSD3(unittest.TestCase):
    @torch.inference_mode
    def test_transformer_sd3(self):
        if not is_test_data_exists():
            return
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["dtype"] = [torch.float16]
        for arg in gen_arg_list(arg_dict):
            for model_path, is_small_fake_model in _get_model_paths():
                _test_transformer_sd3(self, model_path, is_small_fake_model, *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


class TestScheduler(unittest.TestCase):
    @torch.inference_mode
    def test_scheduler(self):
        if not is_test_data_exists():
            return
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["dtype"] = [torch.float16]
        for arg in gen_arg_list(arg_dict):
            for model_path, _ in _get_model_paths():
                _test_scheduler(self, model_path, *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


class TestVAE(unittest.TestCase):
    @torch.inference_mode
    def test_vae(self):
        if not is_test_data_exists():
            return
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["dtype"] = [torch.float16]
        for arg in gen_arg_list(arg_dict):
            for model_path, _ in _get_model_paths():
                _test_vae(self, model_path, *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


class TestPipeline(unittest.TestCase):
    @torch.inference_mode
    def test_pipeline(self):
        if not is_test_data_exists():
            return
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda"]
        arg_dict["dtype"] = [torch.float16]
        for arg in gen_arg_list(arg_dict):
            for model_path, is_small_fake_model in _get_model_paths():
                _test_pipeline(self, model_path, is_small_fake_model, *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


if __name__ == "__main__":
    unittest.main()
