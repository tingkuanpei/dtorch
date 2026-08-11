"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Tuple, Union, List, Optional, Union
import numbers

import torch

import dtorch
from dtorch.distributed_spec import get_default_device_mesh
from dtorch import Tensor, DeviceMesh
from dtorch.nn.parameter import Parameter
from .module import Module
from .. import functional as F

_shape_t = Union[int, List[int], torch.Size]


class LayerNorm(Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        super(LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            factory_kwargs = {
                "dtype": dtype,
                "device_mesh": get_default_device_mesh(device, device_mesh),
            }

            self.weight = Parameter(dtorch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = Parameter(dtorch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class RMSNorm(Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        super(RMSNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            factory_kwargs = {
                "dtype": dtype,
                "device_mesh": get_default_device_mesh(device, device_mesh),
            }

            self.weight = Parameter(dtorch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


class GroupNorm(Module):
    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        super(GroupNorm, self).__init__()

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            factory_kwargs = {
                "dtype": dtype,
                "device_mesh": get_default_device_mesh(device, device_mesh),
            }

            self.weight = Parameter(dtorch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(dtorch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
