"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from dataclasses import dataclass
from typing import Sequence, Optional, Union, Dict
import functools
import importlib

import torch

import dtorch
from dtorch import (
    Tensor,
    Graph,
    DeviceMesh,
    Placement,
    Replicate,
)


# Import all functions from dtorch._dtorch_py_api.nn.functional.
# No Python-side wrapping needed:
# - Input: DTorchTensor/Graph/DeviceMesh/Placement inherit from C++ nanobind wrappers
# - Input: OperatorFormat/PaddingType/PoolingKind have custom type_casters in py_function_type_casters.h
# - Output: generated C++ lambdas call WrapTensor/WrapTensorArray to return DTorchTensor directly
_cpp_module = importlib.import_module("dtorch._dtorch_py_api.nn.functional")
_function_names = [name for name in dir(_cpp_module) if not (name.startswith("__") and name.endswith("__"))]
for function_name in _function_names:
    globals()[function_name] = getattr(_cpp_module, function_name)


# ============================================================
# Tensor Creation Operators
# _zeros, _ones, _full, _empty, _arange, _randint, _rand, _randn, _from_torch
# Decorated by create_op_decorator to auto-handle graph/device/device_mesh args
# ============================================================


def create_op_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        graph = kwargs.pop("graph", None)
        if graph is None:
            graph = Graph.default_graph()

        device = kwargs.pop("device", None)
        device_mesh = kwargs.pop("device_mesh", None)
        assert not (device is not None and device_mesh is not None), "Can't set device and device_mesh at same time"
        if device is not None:
            device_mesh = DeviceMesh(device)
        kwargs["device_mesh"] = device_mesh

        if func.__name__ in {"_zeros", "_empty", "_ones", "_rand", "_randn"}:
            if len(args) == 1:
                if isinstance(args[0], int):
                    shape = list(args)
                else:
                    shape = list(args[0])
            else:
                shape = list(args)
            return func(graph, shape, **kwargs)
        else:
            return func(graph, *args, **kwargs)

    return wrapper


# When the device and device_mesh of the create operator are None, a tensor will be created
# on the graph.default_device_mesh.
_zeros = create_op_decorator(_zeros)
_ones = create_op_decorator(_ones)
_full = create_op_decorator(_full)
_empty = create_op_decorator(_empty)
_arange = create_op_decorator(_arange)
_randint = create_op_decorator(_randint)
_rand = create_op_decorator(_rand)
_randn = create_op_decorator(_randn)
_from_torch = create_op_decorator(_from_torch)


# ============================================================
# _like Creation Operators
# get_like_kwargs extracts input metadata; _zeros_like/_ones_like/_full_like reuse factory ops
# ============================================================


def get_like_kwargs(input, kwargs):
    kwargs["graph"] = kwargs.pop("graph", input.graph)
    kwargs["dtype"] = kwargs.pop("dtype", input.dtype)
    kwargs["device"] = kwargs.pop("device", None)
    kwargs["device_mesh"] = kwargs.pop("device_mesh", None)
    kwargs["placements"] = kwargs.pop("placements", input.placements)
    if kwargs["device"] is None and kwargs["device_mesh"] is None:
        kwargs["device_mesh"] = input.device_mesh
    return input.shape, kwargs


def _zeros_like(input, **kwarg) -> Tensor:
    shape, kwargs = get_like_kwargs(input, kwarg)
    return _zeros(shape, **kwargs)


def _ones_like(input, **kwarg) -> Tensor:
    shape, kwargs = get_like_kwargs(input, kwarg)
    return _ones(shape, **kwargs)


def _full_like(input, fill_value, **kwarg) -> Tensor:
    shape, kwargs = get_like_kwargs(input, kwarg)
    return _full(shape, fill_value, **kwargs)


# ============================================================
# Reduction Operations
# _max, _min: max_min_decorator dispatches between element-wise and reduction modes
# _argmax, _argmin: return indices of extreme values
# ============================================================


def max_min_decorator(func, return_type, element_wise_func=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # If second argument is a Tensor, use element-wise operation (e.g. torch.max(input, other))
        if element_wise_func is not None:
            if len(args) >= 2 and isinstance(args[1], Tensor):
                return element_wise_func(*args, **kwargs)
            if "other" in kwargs and isinstance(kwargs["other"], Tensor):
                other = kwargs.pop("other")
                return element_wise_func(args[0], other, **kwargs)

        outputs = func(*args, **kwargs)
        assert isinstance(outputs, list)
        if len(outputs) == 2:
            return return_type(outputs)
        else:
            return outputs[0]

    return wrapper


_max = max_min_decorator(_max, torch.return_types.max, _maximum)
_min = max_min_decorator(_min, torch.return_types.min, _minimum)


def _argmax(input, *args, **kwargs) -> Tensor:
    if len(args) + len(kwargs) == 0:
        return input.flatten().max(dim=0).indices
    else:
        return input.max(*args, **kwargs).indices


def _argmin(input, *args, **kwargs) -> Tensor:
    if len(args) + len(kwargs) == 0:
        return input.flatten().min(dim=0).indices
    else:
        return input.min(*args, **kwargs).indices


# ============================================================
# Scaled Dot-Product Attention
# SdpaOption: configures Context Parallel / Sage Attention options
# scaled_dot_product_attention: auto-routes to CP variant or standard implementation
# ============================================================


@dataclass
class SdpaOption:
    # cp
    ulysess_cp_dim: Optional[Union[int, str]] = "ulysess_cp"
    ring_cp_dim: Optional[Union[int, str]] = "ring_cp"

    # candidate: "", "auto", "qk_int8_pv_fp8", "qk_int8_pv_fp16"
    sage_attn_type: Optional[str] = None

    def check_valid(self):
        if not self.sage_attn_type in (
            None,
            "",
            "auto",
            "qk_int8_pv_fp8",
            "qk_int8_pv_fp16",
        ):
            return False

        return True


def scaled_dot_product_attention(
    query: Tensor,  # N, Hq, L, E
    key: Tensor,  # N, H, S, E
    value: Tensor,  # N, H, S, E
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa=False,
    sdpa_option: SdpaOption = SdpaOption(),
):
    assert sdpa_option.check_valid(), f"sdpa_option invalid: {sdpa_option}"

    kwarg = {
        "query": query,
        "key": key,
        "value": value,
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
        "enable_gqa": enable_gqa,
        "sdpa_option": sdpa_option,
    }

    if sdpa_option.sage_attn_type:
        try:
            import sageattention
        except ImportError:
            raise RuntimeError("Require sageattention, please install sageattention.")

    if (
        isinstance(sdpa_option.ulysess_cp_dim, int)
        or isinstance(sdpa_option.ring_cp_dim, int)
        or (isinstance(sdpa_option.ulysess_cp_dim, str) and query.device_mesh.has_dim_name(sdpa_option.ulysess_cp_dim))
        or (isinstance(sdpa_option.ring_cp_dim, str) and query.device_mesh.has_dim_name(sdpa_option.ring_cp_dim))
    ):
        from .scaled_dot_product_attention_with_cp import (
            scaled_dot_product_attention_with_cp,
        )

        return scaled_dot_product_attention_with_cp(
            **kwarg,
        )
    else:
        return _scaled_dot_product_attention(
            **kwarg,
        )


# ============================================================
# Graph-Breaking Operators
# These ops call to_torch() to materialize values, forcing synchronization and
# breaking the lazy graph execution. Use sparingly.
# _nonzero, _equal_return_bool, _item, _tolist and tensor.__getitem__
#
# Also note: tensor.__getitem__ with a dtorch.Tensor index triggers a graph break.
# In _parse_index(), the dtorch.Tensor index is passed to dtorch._dtorch_py_api.Index(),
# which internally calls to_torch() to convert it into a torch.Tensor for use as an
# indexing operand. This materialization forces synchronization and breaks the lazy graph.
# ============================================================


def _nonzero(input: Tensor) -> Tensor:
    return Tensor(
        input.to_torch().nonzero(),
        device_mesh=input.device_mesh,
        placements=input.placements,
    )


def _equal_return_bool(input: Tensor, other: Tensor) -> bool:
    if isinstance(input, Tensor) and isinstance(other, Tensor):
        if input.shape != other.shape:
            return False

    result = _equal(input, other)
    torch_result = result.to_torch()
    return bool(torch_result.all().item())


def _item(input: Tensor) -> Union[int, float, bool]:
    return input.to_torch().item()


def _tolist(input: Tensor) -> list:
    return input.to_torch().tolist()


# ============================================================
# Other Operators
# _unbind, _stack, _clip
# ============================================================


def _unbind(input, dim: int = 0) -> Tensor:
    num_chunks = input.shape[dim]
    outputs = input.chunk(num_chunks, dim)
    outputs = [output.squeeze(dim) for output in outputs]
    return outputs


def _stack(tensors, dim: int = 0) -> Tensor:
    tensors = [tensor.unsqueeze(dim) for tensor in tensors]
    output = dtorch.cat(tensors, dim)
    return output


_clip = _clamp
