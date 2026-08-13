"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import sys
import os

import dtorch.util.torch_lib_path
import dtorch._dtorch_py_api

__version__ = dtorch._dtorch_py_api._version
__compile_arguments__ = dtorch._dtorch_py_api._compile_arguments

from dtorch import compiled_op
from dtorch.cluster import MainNode
from dtorch.global_option import GlobalOption

from dtorch.type import OperatorFormat, PaddingType, PoolingKind
from dtorch.memory_stats import MemoryStat, MemoryStats
from dtorch.distributed_spec import (
    DeviceMesh,
    Placement,
    Shard,
    Replicate,
    Partial,
    init_device_mesh,
    get_default_device_mesh,
    assign_layers_to_stages,
)
from dtorch.tensor import Tensor, tensor, TensorFuture
from dtorch.graph import Graph, GraphOption, VoidFutureCollect
from dtorch import nn
from dtorch import cuda
from dtorch.nn.functional import (
    sigmoid,
    tanh,
    relu,
    batch_norm,
    conv2d,
    dropout,
    embedding,
    group_norm,
    layer_norm,
    rms_norm,
)
from dtorch.nn.functional import _matmul as matmul
from dtorch.nn.functional import _reshape as reshape
from dtorch.nn.functional import _concat as cat
from dtorch.nn.functional import _concat as concat
from dtorch.nn.functional import _stack as stack
from dtorch.nn.functional import _abs as abs
from dtorch.nn.functional import _chunk as chunk
from dtorch.nn.functional import _arange as arange
from dtorch.nn.functional import _add as add
from dtorch.nn.functional import _sub as sub
from dtorch.nn.functional import _div as div
from dtorch.nn.functional import _mul as mul
from dtorch.nn.functional import _neg as neg
from dtorch.nn.functional import _reciprocal as reciprocal
from dtorch.nn.functional import _repeat_interleave as repeat_interleave
from dtorch.nn.functional import softmax as softmax
from dtorch.nn.functional import _squeeze as squeeze
from dtorch.nn.functional import _unsqueeze as unsqueeze
from dtorch.nn.functional import _flatten as flatten
from dtorch.nn.functional import _transpose as transpose
from dtorch.nn.functional import _zeros as zeros
from dtorch.nn.functional import _from_torch as from_torch
from dtorch.nn.functional import _zeros_like as zeros_like
from dtorch.nn.functional import _ones as ones
from dtorch.nn.functional import _ones_like as ones_like
from dtorch.nn.functional import _full as full
from dtorch.nn.functional import _full_like as full_like
from dtorch.nn.functional import _empty as empty
from dtorch.nn.functional import _einsum as einsum
from dtorch.nn.functional import _randint as randint
from dtorch.nn.functional import _rand as rand
from dtorch.nn.functional import _randn as randn
from dtorch.nn.functional import _permute as permute
from dtorch.nn.functional import _exp as exp
from dtorch.nn.functional import _log as log
from dtorch.nn.functional import _log2 as log2
from dtorch.nn.functional import _log10 as log10
from dtorch.nn.functional import _isinf as isinf
from dtorch.nn.functional import _isnan as isnan
from dtorch.nn.functional import _square as square
from dtorch.nn.functional import _rsqrt as rsqrt
from dtorch.nn.functional import _sin as sin
from dtorch.nn.functional import _asin as asin
from dtorch.nn.functional import _cos as cos
from dtorch.nn.functional import _floor as floor
from dtorch.nn.functional import _round as round
from dtorch.nn.functional import _pow as pow
from dtorch.nn.functional import _equal_return_bool as equal
from dtorch.nn.functional import _equal as eq
from dtorch.nn.functional import _greater as greater
from dtorch.nn.functional import _greater as gt
from dtorch.nn.functional import _less as less
from dtorch.nn.functional import _less as lt
from dtorch.nn.functional import _greater_equal as greater_equal
from dtorch.nn.functional import _greater_equal as ge
from dtorch.nn.functional import _less_equal as less_equal
from dtorch.nn.functional import _less_equal as le
from dtorch.nn.functional import _logical_and as logical_and
from dtorch.nn.functional import _logical_or as logical_or
from dtorch.nn.functional import _nonzero as nonzero
from dtorch.nn.functional import _clamp as clamp
from dtorch.nn.functional import _clip as clip
from dtorch.nn.functional import _clone as clone
from dtorch.nn.functional import _max as max
from dtorch.nn.functional import _min as min
from dtorch.nn.functional import _minimum as minimum
from dtorch.nn.functional import _maximum as maximum
from dtorch.nn.functional import _sum as sum
from dtorch.nn.functional import _mean as mean
from dtorch.nn.functional import _any as any
from dtorch.nn.functional import _all as all
from dtorch.nn.functional import _where as where
from dtorch.nn.functional import _masked_fill as masked_fill
from dtorch.nn.functional import _masked_scatter as masked_scatter
from dtorch.nn.functional import _unbind as unbind
from dtorch.nn.functional import _argmax as argmax
from dtorch.nn.functional import _argmin as argmin
from dtorch.nn.functional import _outer as outer
from dtorch.nn.functional import SdpaOption

default_graph = Graph.default_graph()


def get_data_type_tensor(data_type):
    class DataTypeTensor(dtorch.Tensor):
        def __instancecheck__(cls, instance):
            return instance.dtype == data_type

    return DataTypeTensor


import torch
import numpy as np

dtype = torch.dtype
finfo = torch.finfo
iinfo = torch.iinfo
device = torch.device
float32 = torch.float32
float = torch.float
float64 = torch.float64
float16 = torch.float16
bfloat16 = torch.bfloat16
uint8 = torch.uint8
int8 = torch.int8
int16 = torch.int16
short = torch.int16
int32 = torch.int32
int = torch.int32
int64 = torch.int64
long = torch.int64
bool = torch.bool

FloatTensor = get_data_type_tensor(torch.float32)
DoubleTensor = get_data_type_tensor(torch.float64)
HalfTensor = get_data_type_tensor(torch.float16)
BFloat16Tensor = get_data_type_tensor(torch.bfloat16)
ByteTensor = get_data_type_tensor(torch.uint8)
CharTensor = get_data_type_tensor(torch.int8)
ShortTensor = get_data_type_tensor(torch.int16)
IntTensor = get_data_type_tensor(torch.int)
LongTensor = get_data_type_tensor(torch.long)
BoolTensor = get_data_type_tensor(torch.bool)


def from_numpy(ndarray: np.ndarray) -> dtorch.Tensor:
    return dtorch.Tensor(torch.from_numpy(ndarray))


Generator = torch.Generator
Size = torch.Size
