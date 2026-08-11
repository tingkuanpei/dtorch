"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import itertools
from collections import OrderedDict
from typing import Union
import os
from dataclasses import dataclass
import math

import numpy as np
import torch

import dtorch
from dtorch import Graph, DeviceMesh
from dtorch.distributed_spec import init_device_mesh
from dtorch.util.tensor_checker import TensorChecker


def get_dtorch_python_root_path():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))


def get_test_data_path():
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "../", "../", "test_data")
    return os.environ.get("DTORCH_TEST_DATA_DIR", default_path)


def is_test_data_exists():
    return os.path.exists(os.path.join(get_test_data_path(), "dtorch_test_data"))


def check_pytorch_version(min_version, max_version=None):
    current_version = torch.__version__.split("+")[0]

    current_version_parts = tuple(map(int, current_version.split(".")))
    min_version_parts = tuple(map(int, min_version.split(".")))
    if max_version is not None:
        max_version_parts = tuple(map(int, max_version.split(".")))

    if current_version_parts < min_version_parts:
        return False

    if max_version is not None and current_version_parts > max_version_parts:
        return False

    return True


def gen_arg_list(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    assert all([isinstance(x, list) for x in arg_dict.values()])

    for (key, value_list) in arg_dict.items():
        if key == "device":
            remove_cuda = not torch.cuda.is_available()
            if remove_cuda:
                value_list = [value for value in value_list if value != "cuda"]
            arg_dict[key] = value_list

    sets = [arg_set for (_, arg_set) in arg_dict.items()]
    return itertools.product(*sets)


def assert_tensor_meanclose(
    test_case,
    a,
    b,
    rtol: float = 1e-05,
    atol: float = 1e-08,
):
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple))
        test_case.assertTrue(len(a) == len(b))
        for it_a, it_b in zip(a, b):
            assert_tensor_meanclose(test_case, it_a, it_b)
    else:
        test_case.assertTrue(
            TensorChecker.tensor_meanclose(
                a=a,
                b=b,
                rtol=rtol,
                atol=atol,
                print_no_equal_msg=True,
            )
        )


def assert_tensor_equal(test_case, a, b):
    """Assert two tensors are exactly equal (same size and elements).

    Uses torch.Tensor.equal() / dtorch.Tensor.equal() semantics.
    Supports torch.Tensor, dtorch.Tensor, and list/tuple of tensors.
    """
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple))
        test_case.assertTrue(len(a) == len(b))
        for it_a, it_b in zip(a, b):
            assert_tensor_equal(test_case, it_a, it_b)
    else:
        test_case.assertTrue(TensorChecker.tensor_equal(a=a, b=b, print_no_equal_msg=True))


def assert_tensor_allclose(
    test_case,
    a,
    b,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
):
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple))
        test_case.assertTrue(len(a) == len(b))
        for it_a, it_b in zip(a, b):
            assert_tensor_allclose(test_case, it_a, it_b)
    else:
        test_case.assertTrue(
            TensorChecker.tensor_allclose(
                a=a,
                b=b,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                print_no_equal_msg=True,
            )
        )


def print_all_gpu_memory(reset_peak_stats: bool = False):
    print("-" * 50)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_mesh = init_device_mesh("cuda", num_gpus)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_stats = dtorch.default_graph.get_memory_stats(device_mesh=device_mesh, reset_peak=reset_peak_stats)
        print(f"Detected {num_gpus} available GPUs. Memory per device: {total_memory / 1024**2:.2f} MB")
        print(memory_stats)
    else:
        print("CUDA is not available.")
    print("-" * 50)


@dataclass
class MemoryRequire:
    parameter_gb: int
    same_device_parameter_plus_gb: int = 0
    activation_gb: int = 4
    same_device_activation_plus_gb: int = 1


def is_graph_satisfy(graph: Graph, device_mesh: DeviceMesh = None, memory_require: MemoryRequire = None):
    if device_mesh is not None and not graph.satisfy(device_mesh):
        return False

    if device_mesh is not None and memory_require is not None:
        tp_size = 1
        tp_dim = device_mesh.dim_name_index("tp")
        if tp_dim is not None:
            tp_size = device_mesh.shape[tp_dim]
        if tp_size > 1:
            memory_require.parameter_gb = memory_require.parameter_gb / tp_size * 1.1

    if memory_require is not None:
        if device_mesh is not None and device_mesh.device_type == torch.device("cpu"):
            return True

        if dtorch.GlobalOption.get_dtensor_in_same_device():
            if device_mesh is None:
                device_count = 1
            else:
                device_count = math.prod(device_mesh.shape)
            require_size = (
                memory_require.parameter_gb * device_count
                + memory_require.activation_gb * device_count
                + (memory_require.same_device_activation_plus_gb + memory_require.same_device_parameter_plus_gb)
                * (device_count - 1)
            )
        else:
            require_size = memory_require.parameter_gb + memory_require.activation_gb

        assert torch.cuda.is_available()
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        if require_size > total_memory_gb:
            return False

    return True
