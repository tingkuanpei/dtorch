"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Optional, Tuple, Union
import math

import dtorch
from dtorch import (
    Tensor,
    PoolingKind,
    PaddingType,
    OperatorFormat,
)
from .module import Module
from ..common_types import _size_2_t, _size_4_t
from .. import functional as F


class Pooling2d(Module):
    pooling_Kind: PoolingKind
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    ceil_mode: bool
    count_include_pad: bool
    is_nchw: bool

    def __init__(
        self,
        pooling_Kind: PoolingKind,
        kernel_size: _size_2_t,
        *,
        stride: _size_2_t = (1, 1),
        padding: Union[str, _size_2_t, _size_4_t] = (0, 0),
        dilation: _size_2_t = (1, 1),
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        is_nchw: bool = True,
    ) -> None:
        super(Pooling2d, self).__init__()

        self.pooling_Kind = pooling_Kind
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

        # pad
        self.padding_type = PaddingType.not_set
        self.actual_pad = (0, 0)
        if isinstance(padding, str):
            if padding == "same":
                self.padding_type = PaddingType.same
            elif padding == "valid":
                self.padding_type = PaddingType.valid
            else:
                raise ValueError("Invalid padding string {!r}, should be one of { same, valid }".format(padding))
        else:
            self.actual_pad = padding

        # operator format
        self.operator_format = OperatorFormat.nchw
        if not is_nchw:
            self.operator_format = OperatorFormat.nhwc

    def forward(self, input: Tensor) -> Tensor:
        return F._pooling2d(
            input,
            self.pooling_Kind,
            self.dilation,
            self.kernel_size,
            self.stride,
            self.padding_type,
            self.actual_pad,
            self.ceil_mode,
            self.count_include_pad,
            self.operator_format,
        )


class AvgPool2d(Pooling2d):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = (1, 1),
        padding: _size_2_t = (0, 0),
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        dilation: _size_2_t = (1, 1),
        is_nchw: bool = True,
    ):
        super(AvgPool2d, self).__init__(
            PoolingKind.avg,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            is_nchw=is_nchw,
        )


class MaxPool2d(Pooling2d):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = (1, 1),
        padding: _size_2_t = (0, 0),
        dilation: _size_2_t = 1,
        ceil_mode: bool = False,
        is_nchw: bool = True,
    ):
        super(MaxPool2d, self).__init__(
            PoolingKind.max,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            is_nchw=is_nchw,
        )


class GlobalPool2d(Module):
    pooling_Kind: PoolingKind
    is_nchw: bool

    def __init__(
        self,
        pooling_Kind: PoolingKind,
        is_nchw: bool = True,
    ) -> None:
        super(GlobalPool2d, self).__init__()

        self.pooling_Kind = pooling_Kind

        # operator format
        self.operator_format = OperatorFormat.nchw
        if not is_nchw:
            self.operator_format = OperatorFormat.nhwc

    def forward(self, input: Tensor) -> Tensor:
        return F._global_pooling2d(input, self.pooling_Kind, self.operator_format)


class GlobalAvgPool2d(GlobalPool2d):
    def __init__(
        self,
        is_nchw: bool = True,
    ):
        super(GlobalAvgPool2d, self).__init__(
            PoolingKind.avg,
            is_nchw=is_nchw,
        )


class GlobalMaxPool2d(GlobalPool2d):
    def __init__(
        self,
        is_nchw: bool = True,
    ):
        super(GlobalMaxPool2d, self).__init__(
            PoolingKind.max,
            is_nchw=is_nchw,
        )


# adaptive average pool to average pool
# https://github.com/pytorch/pytorch/blob/65b00aa5972e23b2a70aa60dec5125671a3d7153/aten/src/ATen/native/AdaptiveAveragePooling.cpp
# https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
#
# 绝大多数高性能加速库，都不支持 PyTorch 的 AdaptiveAvgPoolNd 和 AdaptiveMaxPoolNd，其计算过程可以用 AvgPoolNd 和 MaxPoolNd
# 替代，但是计算过程的细微差异，无法和 PyTorch 的实现完全对齐。因此需要让用户获知这一差异。
class AdaptivePool:
    @staticmethod
    def to_normal_pool(input_size: int, output_size: int):
        stride = math.floor(input_size / output_size)
        kernel = input_size - (output_size - 1) * stride
        return (stride, kernel)
