"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Optional, Tuple, Union

import torch

import dtorch
from dtorch import Tensor, PaddingType, OperatorFormat, DeviceMesh, Replicate
from dtorch.distributed_spec import get_default_device_mesh
from dtorch.nn.parameter import Parameter
from .module import Module
from .utils import _pair
from ..common_types import _size_2_t, _size_4_t
from .. import functional as F


class _ConvNd(Module):
    pass


class Conv2d(Module):
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    groups: int
    padding_type: PaddingType
    actual_pad: Tuple[int, ...]
    weight: Tensor
    bias: Optional[Tensor]
    operator_format: OperatorFormat

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = (1, 1),
        padding: Union[str, _size_2_t, _size_4_t] = (0, 0),
        dilation: _size_2_t = (1, 1),
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        is_nchw: bool = True,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        super(Conv2d, self).__init__()

        # group
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        # pad
        padding_type = PaddingType.not_set
        actual_pad = (0, 0)
        if isinstance(padding, str):
            if padding == "same":
                padding_type = PaddingType.same
            elif padding == "valid":
                padding_type = PaddingType.valid
            else:
                raise ValueError("Invalid padding string {!r}, should be one of { same, valid }".format(padding))
        else:
            actual_pad = padding

        # operator format
        operator_format = OperatorFormat.nchw
        if not is_nchw:
            operator_format = OperatorFormat.nhwc

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_type = padding_type
        self.actual_pad = actual_pad
        self.operator_format = operator_format

        factory_kwargs = {
            "dtype": dtype,
            "device_mesh": get_default_device_mesh(device, device_mesh),
        }
        self.weight = Parameter(dtorch.empty(out_channels, in_channels // groups, *self.kernel_size, **factory_kwargs))

        if bias:
            self.bias = Parameter(dtorch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding_type,
            self.actual_pad,
            self.dilation,
            self.groups,
            self.operator_format,
        )
