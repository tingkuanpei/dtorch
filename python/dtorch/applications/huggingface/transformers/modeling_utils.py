"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import json
import os
from typing import Type, Union
from functools import wraps

import torch
from transformers.modeling_utils import PreTrainedModel as TransformersPreTrainedModel
from transformers.modeling_utils import SpecificPreTrainedModelType
from safetensors.torch import load_file

import dtorch


def restore_default_torch_dtype(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        old_dtype = dtorch.default_graph.default_dtype
        try:
            return func(*args, **kwargs)
        finally:
            dtorch.default_graph._set_default_dtype(old_dtype)

    return _wrapper


class PreTrainedModel(TransformersPreTrainedModel, dtorch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        dtype_orig = super()._set_default_torch_dtype(dtype)
        dtorch.default_graph._set_default_dtype(dtype)
        return dtype_orig

    @classmethod
    @restore_default_torch_dtype
    def from_pretrained(
        cls,
        *args,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        device_mesh = kwargs.pop("device_mesh", None)
        with dtorch.Graph.default_graph().device_mesh_guard(device_mesh):
            result = super().from_pretrained(*args, **kwargs)
        return result
