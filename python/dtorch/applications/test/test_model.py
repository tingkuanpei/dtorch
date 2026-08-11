"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import os

# In diffusion models, the text_encoder input shapes are small, but the number of kernels launched for
# each text_encoder is large. Setting CUDA_SCALE_LAUNCH_QUEUES=4x prevents the GPU from stalling while waiting for
# the CPU to launch kernels.
os.environ["CUDA_SCALE_LAUNCH_QUEUES"] = "4x"

import argparse
import asyncio

import torch
import numpy as np
from diffusers import FluxPipeline as DiffusersFluxPipeline
from diffusers import StableDiffusion3Pipeline as DiffusersStableDiffusion3Pipeline

import dtorch
from dtorch.applications.huggingface.diffusers import (
    FluxPipeline as DTorchFluxPipeline,
    StableDiffusion3Pipeline as DTorchStableDiffusion3Pipeline,
)
from dtorch.test.test_util import (
    get_test_data_path,
    print_all_gpu_memory,
    is_graph_satisfy,
    MemoryRequire,
)
from dtorch.distributed_spec import init_device_mesh
from dtorch.applications.core.config import (
    ExecuteConfig,
    FirstBlockCacheConfig,
    FuseKernelConfig,
    QuantizeConfig,
)
from dtorch.util.tensor_checker import TensorChecker
from dtorch.util.benchmark import Benchmark
from dtorch.cuda.nvtx import module_register_nvtx

# ---------------------------------------------------------------------------
# Model registry — add new models here
# ---------------------------------------------------------------------------
# Each entry defines:
#   diffusers_cls   : original diffusers pipeline class
#   dtorch_cls      : DTorch-wrapped pipeline class
#   model_dir       : directory name under test_data
#   prompt          : default prompt for image generation
#   pipe_kwargs     : model-specific kwargs forwarded to pipe()
#                     (negative_prompt and generator are added automatically)
#   memory_require  : MemoryRequire(parameter_gb, same_device_parameter_plus_gb)
#   test_configs    : list of dicts, each with keys:
#                       dp, ulysess_cp, ring_cp (default 1)
#                       first_block_cache, sage_attn, fuse_kernel (default False)

MODEL_REGISTRY = {
    "flux": {
        "diffusers_cls": DiffusersFluxPipeline,
        "dtorch_cls": DTorchFluxPipeline,
        "model_dir": "FLUX.1-dev",
        "prompt": "A cat holding a sign that says hello world",
        "pipe_kwargs": {
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 24,
            "max_sequence_length": 512,
        },
        "memory_require": MemoryRequire(parameter_gb=32, same_device_parameter_plus_gb=23),
        "test_configs": [
            {},
            {"first_block_cache": True},
            {"sage_attn": False},
            {"fuse_kernel": True},
            {"ulysess_cp": 2},
            {"ulysess_cp": 2, "first_block_cache": True},
            {"ulysess_cp": 2, "sage_attn": False},
            {"ring_cp": 2},
            {"ulysess_cp": 2, "ring_cp": 2},
        ],
    },
    "sd3": {
        "diffusers_cls": DiffusersStableDiffusion3Pipeline,
        "dtorch_cls": DTorchStableDiffusion3Pipeline,
        "model_dir": "stable-diffusion-3-medium-diffusers",
        "prompt": "A cat holding a sign that says hello world",
        "pipe_kwargs": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 28,
            "guidance_scale": 7.0,
        },
        "memory_require": MemoryRequire(parameter_gb=15, same_device_parameter_plus_gb=4),
        "test_configs": [
            {},
            {"dp": 2},
            {"ulysess_cp": 2},
            {"ring_cp": 2},
            {"ulysess_cp": 2, "ring_cp": 2},
            {"dp": 2, "ulysess_cp": 2, "ring_cp": 2},
        ],
    },
}


def get_model_config(model_name: str):
    """Retrieve the configuration dict for *model_name*.

    Raises KeyError with a list of available models if unknown.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name]


def torch_test(
    config: dict,
    model_path: str,
    prompt: str,
    output_prefix: str,
    generator_seed: int = 0,
    torch_compile: bool = False,
    benchmark: bool = False,
    benchmark_count: int = 3,
):
    """Run the reference (diffusers) pipeline.

    Always returns ``(torch_array, title, metric)``.  *title* and *metric* are
    ``None`` unless *benchmark* is True.
    """
    pipe = config["diffusers_cls"].from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    # module_register_nvtx(pipe.text_encoder, tag="torch_text_encoder")
    # module_register_nvtx(pipe.text_encoder_2, tag="torch_text_encoder_2")
    # module_register_nvtx(pipe.text_encoder_3, tag="torch_text_encoder_3")
    # module_register_nvtx(pipe.transformer, tag="torch_transformer")
    # module_register_nvtx(pipe.vae, tag="torch_vae")

    if torch_compile:
        # "max-autotune-no-cudagraphs"
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

    compile_str = "_compile" if torch_compile else ""
    title = f"torch_{output_prefix}{compile_str}"

    @torch.inference_mode()
    def func():
        return pipe(
            prompt,
            negative_prompt="",
            generator=torch.Generator("cuda").manual_seed(generator_seed),
            **config["pipe_kwargs"],
        ).images[0]

    if benchmark:
        metric, image = Benchmark.run(func, count=benchmark_count)
    else:
        metric = None
        image = func()

    file_path = f"./{title}.jpg"
    image.save(file_path)
    torch_array = np.asarray(image)

    del pipe
    dtorch.default_graph.empty_cache()

    return torch_array, title, metric


def dtorch_test(
    config: dict,
    model_path: str,
    torch_array: np.ndarray,
    prompt: str,
    output_prefix: str,
    dp: int = 1,
    ulysess_cp: int = 1,
    ring_cp: int = 1,
    test_count: int = 1,
    check_output: bool = False,
    first_block_cache: bool = False,
    sage_attn: bool = False,
    fuse_kernel: bool = False,
    generator_seed: int = 0,
    benchmark: bool = False,
    benchmark_count: int = 3,
):
    """Run the DTorch-wrapped pipeline and compare against the reference.

    Always returns ``(title, metric)``.  Both are ``None`` unless *benchmark* is
    True (or when the graph is unsatisfiable).
    """
    device_mesh = init_device_mesh(
        "cuda",
        (dp, ulysess_cp, ring_cp),
        mesh_dim_names=["dp", "ulysess_cp", "ring_cp"],
    )
    if not is_graph_satisfy(dtorch.default_graph, device_mesh, config["memory_require"]):
        return None, None

    execute_config = ExecuteConfig(
        cache_config=FirstBlockCacheConfig(residual_diff_threshold=0.12) if first_block_cache else None,
        quant_config=QuantizeConfig(sage_attn_type="auto") if sage_attn else QuantizeConfig(),
        fuse_kernel_config=FuseKernelConfig(
            fuse_apply_rotary_emb=fuse_kernel,
            fuse_silu_linear_chunk=fuse_kernel,
            fuse_layer_norm_mul_add=fuse_kernel,
        ),
    )
    pipe = config["dtorch_cls"].from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_mesh=device_mesh,
        execute_config=execute_config,
    )
    # module_register_nvtx(pipe.text_encoder, tag="dtorch_text_encoder")
    # module_register_nvtx(pipe.text_encoder_2, tag="dtorch_text_encoder_2")
    # module_register_nvtx(pipe.text_encoder_3, tag="dtorch_text_encoder_3")
    # module_register_nvtx(pipe.transformer, tag="dtorch_transformer")
    # module_register_nvtx(pipe.vae, tag="dtorch_vae")

    dp_str = f"_dp{dp}" if dp > 1 else ""
    ulysess_str = f"_ulysess{ulysess_cp}" if ulysess_cp > 1 else ""
    ring_str = f"_ring{ring_cp}" if ring_cp > 1 else ""
    cache_str = "_first_block_cache" if first_block_cache else ""
    sage_str = "_sage_attn" if sage_attn else ""
    fuse_str = "_fuse_kernel" if fuse_kernel else ""
    title = f"dtorch_{output_prefix}{dp_str}{ulysess_str}{ring_str}{cache_str}{sage_str}{fuse_str}"

    for i in range(test_count):
        image_tensor = pipe(
            prompt,
            negative_prompt="",
            generator=torch.Generator("cuda").manual_seed(generator_seed),
            output_type="pt",
            **config["pipe_kwargs"],
        ).images
        image_future = image_tensor.to_torch_async()

        file_path = f"./{title}_iter_{i}.jpg"
        image = pipe.image_processor.pt_to_pil(image_future.get())[0]
        image.save(file_path)

        dtorch_array = np.asarray(image)
        if check_output and torch_array is not None:
            assert TensorChecker.tensor_meanclose(torch_array, dtorch_array, rtol=0.05)

    last_postprocess_task = None

    async def func(benchmark_last_round=False):
        nonlocal last_postprocess_task

        image_tensor = await pipe.async_call(
            prompt,
            negative_prompt="",
            generator=torch.Generator("cuda").manual_seed(generator_seed),
            output_type="pt",
            **config["pipe_kwargs"],
        )
        image_future = image_tensor.images.to_torch_async()

        if last_postprocess_task is not None:
            await last_postprocess_task

        async def postprocess(image_future):
            image = await image_future
            image = pipe.image_processor.pt_to_pil(image)[0]
            return image

        last_postprocess_task = asyncio.create_task(postprocess(image_future))

        if benchmark_last_round:
            return await last_postprocess_task

    if benchmark:
        metric, image = Benchmark.run(func, count=benchmark_count)

        file_path = f"./{title}_benchmark.jpg"
        image.save(file_path)
    else:
        metric = None

    del pipe
    dtorch.default_graph.empty_cache()

    return title, metric


def run_model(model_name: str, args):
    """Run torch_test and all configured dtorch_test variants for one model."""
    config = get_model_config(model_name)
    model_path = os.path.join(get_test_data_path(), config["model_dir"])

    prompt = args.prompt if args.prompt is not None else config["prompt"]
    titles = []
    metrics = []
    check_output = args.check_output

    if args.no_torch:
        if check_output:
            print("Warning: --no-torch is set, disabling --check-output (no reference available).")
            check_output = False
        torch_array = None
    else:
        compile_str = " with torch.compile" if args.torch_compile else ""
        print(f"Running torch reference for '{model_name}'{compile_str}...")
        torch_array, torch_title, torch_metric = torch_test(
            config,
            model_path,
            prompt,
            model_name,
            generator_seed=args.generator_seed,
            torch_compile=args.torch_compile,
            benchmark=args.benchmark,
            benchmark_count=args.benchmark_count,
        )
        titles.append(torch_title)
        metrics.append(torch_metric)

    for tc in config["test_configs"]:
        dp = tc.get("dp", 1)
        ulysess_cp = tc.get("ulysess_cp", 1)
        ring_cp = tc.get("ring_cp", 1)
        first_block_cache = tc.get("first_block_cache", False)
        sage_attn = tc.get("sage_attn", False)
        fuse_kernel = tc.get("fuse_kernel", False)

        extra = []
        if first_block_cache:
            extra.append("first_block_cache")
        if sage_attn:
            extra.append("sage_attn")
        if fuse_kernel:
            extra.append("fuse_kernel")
        extra_str = f" [{', '.join(extra)}]" if extra else ""

        print(
            f"Running dtorch test for '{model_name}' "
            f"(dp={dp}, ulysess_cp={ulysess_cp}, ring_cp={ring_cp}){extra_str}..."
        )
        dtorch_title, dtorch_metric = dtorch_test(
            config,
            model_path,
            torch_array,
            prompt,
            model_name,
            dp=dp,
            ulysess_cp=ulysess_cp,
            ring_cp=ring_cp,
            test_count=args.test_count,
            check_output=check_output,
            first_block_cache=first_block_cache,
            sage_attn=sage_attn,
            fuse_kernel=fuse_kernel,
            generator_seed=args.generator_seed,
            benchmark=args.benchmark,
            benchmark_count=args.benchmark_count,
        )

        if dtorch_title is not None:
            titles.append(dtorch_title)
            metrics.append(dtorch_metric)

    if args.benchmark:
        Benchmark.print(titles=titles, metrics=metrics)

    print(f"All tests for '{model_name}' passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run DTorch diffusion model tests.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="sd3",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Which model to test (default: sd3). Use 'all' to run every registered model.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the model's default prompt for image generation.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=1,
        help="Number of iterations for each dtorch test configuration (default: 1).",
    )
    parser.add_argument(
        "--check-output",
        action="store_true",
        default=False,
        help="Assert DTorch output matches the torch reference via TensorChecker.",
    )
    parser.add_argument(
        "--generator-seed",
        type=int,
        default=0,
        help="Manual seed for the CUDA generator (default: 0).",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help="Apply torch.compile to the transformer in the torch reference pipeline.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run Benchmark on each pipeline and print a summary table.",
    )
    parser.add_argument(
        "--benchmark-count",
        type=int,
        default=3,
        help="Number of benchmark iterations (overrides Benchmark.run count parameter, default: 3).",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        default=False,
        help="Skip the PyTorch reference run (only run DTorch tests).",
    )
    args = parser.parse_args()

    if args.model == "all":
        for name in MODEL_REGISTRY:
            run_model(name, args)
    else:
        run_model(args.model, args)


if __name__ == "__main__":
    main()
