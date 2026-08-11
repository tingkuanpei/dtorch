"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Sequence, overload, Union, Optional, Tuple, List, Dict
from types import EllipsisType
from functools import reduce
import numbers
import asyncio

import numpy as np
import torch

import dtorch
import dtorch
from dtorch import DeviceMesh, Placement
from dtorch.distributed_spec import (
    get_default_device_mesh,
    get_placements_from_dict,
    Replicate,
)


class DTorchTensor(dtorch._dtorch_py_api.Tensor):
    """Python-side distributed tensor that inherits from the C++ Tensor nanobind wrapper.

    DTorchTensor IS-A C++ Tensor, so nanobind functions can extract the C++ pointer directly
    from the parent class holder without any Python-side unwrapping.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.__init__

        # Path 1: Wrap existing C++ or Python Tensor via copy constructor.
        if (
            len(args) == 1
            and len(kwargs) == 0
            and (isinstance(args[0], dtorch._dtorch_py_api.Tensor) or isinstance(args[0], Tensor))
        ):
            super().__init__(args[0])
            return

        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            torch_tensor = args[0]
        else:
            torch_tensor = torch.Tensor(*args, **kwargs)

        device = kwargs.pop("device", None)
        device_mesh = kwargs.pop("device_mesh", None)
        assert not (device is not None and device_mesh is not None), "Can't set device and device_mesh at same time"
        if device is not None:
            device_mesh = DeviceMesh(device)
        kwargs["device_mesh"] = device_mesh

        graph = kwargs.pop("graph", None)
        if graph is None:
            from dtorch import Graph

            graph = Graph.default_graph()

        if device_mesh is not None:
            torch_tensor = torch_tensor.to(device=device_mesh.device_type)

        # When the torch_tensor is a CUDA tensor, since dtorch uses other CUDA streams, it is necessary to synchronize
        # the PyTorch CUDA current_stream first. If multiple CUDA stream environments are used in PyTorch, users need
        # to ensure that the Torch tensors are already synchronized.
        if torch_tensor.is_cuda:
            torch.cuda.current_stream().synchronize()

        super().__init__(graph, torch_tensor, **kwargs)

    def __repr__(self) -> str:
        return (
            f"dtorch {self.to_torch()}\nshape={self.shape}, dtype={self.dtype},\n"
            f"device_mesh={self.device_mesh}, placements={self.placements}"
        )

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._get_shape())

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        return reduce(lambda x, y: x * y, self.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._get_dtype()

    @property
    def is_dtensor(self) -> bool:
        return self._is_distributed()

    @property
    def device(self) -> torch.device:
        if self.device_mesh.is_distributed:
            return self.device_mesh.device_type
        else:
            return self.device_mesh.to_device()

    @property
    def device_mesh(self) -> DeviceMesh:
        return DeviceMesh(self._get_default_device_mesh())

    @property
    def placements(self) -> Sequence[Placement]:
        return [Placement(it) for it in self._get_placement_seq()]

    def check_placement(
        self,
        dim_idx: int,
        expect_placement: Placement,
        skip_when_dim_size_equal_one: bool = True,
    ) -> bool:
        device_mesh = self.device_mesh
        if skip_when_dim_size_equal_one and device_mesh.shape[dim_idx] == 1:
            return True

        placement = self.placements[dim_idx]
        return placement == expect_placement

    @property
    def data(self) -> "DTorchTensor":
        return self

    @data.setter
    def data(self, value: "DTorchTensor") -> None:
        # Mirror PyTorch's `param.data = new_value` by swapping the internal C++ Impl.
        # InplaceAssignment swaps the shared_ptr<Impl>, so self shares the same
        # underlying Operand/Graph state as value. Python object identity is preserved
        # (critical for nn.Module._parameters), while dtype/shape/data are fully replaced.
        self._inplace_assignment(value)

    @property
    def graph(self) -> "dtorch.Graph":
        return dtorch.Graph(self._get_graph())

    def to_torch(self) -> torch.Tensor:
        return self._get_torch_tensor()

    def to_torch_async(self) -> "TensorFuture":
        """Asynchronously get the Tensor value, immediately returns TensorFuture.

        Compared to to_torch():
        - to_torch(): synchronous blocking, waits until tensor value is computed
        - to_torch_async(): immediately returns TensorFuture, user can call future.get() later to get the value
        """
        return TensorFuture(self._get_torch_tensor_async())

    def numpy(self) -> np.array:
        return self.to_torch().numpy()

    def dim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        shape = self.shape
        if len(shape) <= 0:
            raise TypeError("len() of a 0-d tensor")
        return shape[0]

    def __hash__(self):
        return id(self)

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        shape = self._get_shape()
        if dim is None:
            return torch.Size(shape)
        # Support negative indexing, aligning with torch.Tensor.size(dim)
        if dim < 0:
            dim += len(shape)
        return shape[dim]

    @staticmethod
    def parser_tensor_to_function_arg(args, kwargs):
        device_mesh = None
        dtype = None
        error_str = f"Invalid argument, args: {args}, kwargs: {kwargs}"

        def check_is_none(value):
            if value is not None:
                raise KeyError(error_str)

        for arg in args:
            if isinstance(arg, torch.dtype):
                check_is_none(dtype)
                dtype = arg
            elif isinstance(arg, torch.device) or isinstance(arg, str) or isinstance(arg, dtorch.DeviceMesh):
                check_is_none(device_mesh)
                device_mesh = dtorch.DeviceMesh(arg)
            elif isinstance(arg, dtorch.Tensor):
                check_is_none(dtype)
                check_is_none(device_mesh)
                dtype = arg.dtype
                device_mesh = arg.device_mesh
            else:
                raise KeyError(error_str)

        for key, value in kwargs.items():
            if value is None:
                continue
            if key == "device" or key == "device_mesh":
                check_is_none(device_mesh)
                device_mesh = dtorch.DeviceMesh(value)
            elif key == "dtype":
                check_is_none(dtype)
                assert isinstance(value, torch.dtype)
                dtype = value
            else:
                raise KeyError(error_str)

        return device_mesh, dtype

    def to(self, *args, **kwargs) -> "Tensor":
        device_mesh, dtype = self.parser_tensor_to_function_arg(args, kwargs)

        if device_mesh is not None and device_mesh.shape != self.device_mesh.shape:
            raise KeyError(
                f"The device_mesh in the parameter and the device_mesh of the input tensor have different shapes, "
                f"from {self.device_mesh.shape} vs to {device_mesh.shape}. "
                f"If you need to convert the device_mesh of a tensor, use the tensor.redistribute() instead."
            )

        return dtorch.nn.functional._to(self, device_mesh=device_mesh, dtype=dtype)

    def cpu(self) -> "Tensor":
        return self.to(device="cpu")

    def cuda(self) -> "Tensor":
        return self.to(device="cuda")

    def detach(self) -> "Tensor":
        # DTorch not support gradient，so return directly
        return self

    def type_as(self, tensor: "Tensor") -> "Tensor":
        return self.to(dtype=tensor.dtype)

    def is_floating_point(self) -> bool:
        return self.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )

    def half(self) -> "Tensor":
        return self.to(torch.float16)

    def bfloat16(self) -> "Tensor":
        return self.to(torch.bfloat16)

    def float(self) -> "Tensor":
        return self.to(torch.float32)

    def double(self) -> "Tensor":
        return self.to(torch.float64)

    def bool(self) -> "Tensor":
        return self.to(torch.bool)

    def byte(self) -> "Tensor":
        return self.to(torch.uint8)

    def char(self) -> "Tensor":
        return self.to(torch.int8)

    def short(self) -> "Tensor":
        return self.to(torch.int16)

    def int(self) -> "Tensor":
        return self.to(torch.int32)

    def long(self) -> "Tensor":
        return self.to(torch.int64)

    def item(self):
        return dtorch.nn.functional._item(self)

    def tolist(self):
        return dtorch.nn.functional._tolist(self)

    def copy_(self, other: "Tensor") -> "Tensor":
        dtorch.nn.functional._copy(self, other)
        return self

    def clone(self) -> "Tensor":
        return dtorch.nn.functional._clone(self)

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
    ) -> "Tensor":
        assert device_mesh is not None or placements is not None
        return dtorch.nn.functional._redistribute(
            self,
            device_mesh=device_mesh,
            placements=placements,
        )

    def redistribute_like(
        self,
        tensor: "Tensor",
    ) -> "Tensor":
        return self.redistribute(device_mesh=tensor.device_mesh, placements=tensor.placements)

    def redistribute_by_dict(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements_dict: Optional[Dict[Union[str, int], Placement]] = None,
        *,
        default_placement_mode: str = "raise_error",
        convert_shard_size_one_to_replicate: bool = True,
    ) -> "Tensor":
        if placements_dict is not None and len(placements_dict) == 0:
            placements_dict = None

        device_mesh = device_mesh if device_mesh is not None else self.device_mesh

        # from placements_dict
        if placements_dict is not None:
            if device_mesh.is_distributed:
                placements = get_placements_from_dict(
                    device_mesh=device_mesh,
                    placements_dict=placements_dict,
                    original_placements=self.placements,
                    default_placement_mode=default_placement_mode,
                )
            else:
                placements = [Replicate()] * device_mesh.ndim
        else:
            placements = self.placements

        # convert_shard_size_one_to_replicate
        if convert_shard_size_one_to_replicate:
            placements = [
                Replicate() if placement.is_shard() and self.shape[placement.get_shard_index()] == 1 else placement
                for placement in placements
            ]

        return self.redistribute(device_mesh=device_mesh, placements=placements)

    def matmul(self, other: "Tensor") -> "Tensor":
        return dtorch.matmul(self, other)

    def add(self, other: Union[double, "Tensor"]) -> "Tensor":
        return dtorch.add(self, other)

    def sub(self, other: Union[double, "Tensor"]) -> "Tensor":
        return dtorch.sub(self, other)

    def mul(self, other: Union[double, "Tensor"]) -> "Tensor":
        return dtorch.mul(self, other)

    def div(self, other: Union[double, "Tensor"]) -> "Tensor":
        return dtorch.div(self, other)

    def neg(self) -> "Tensor":
        return dtorch.neg(self)

    def reciprocal(self) -> "Tensor":
        return dtorch.reciprocal(self)

    def pow(self, exponent: Union[float, "Tensor"]) -> "Tensor":
        return dtorch.pow(self, exponent)

    def exp(self) -> "Tensor":
        return dtorch.exp(self)

    def log(self) -> "Tensor":
        return dtorch.log(self)

    def log2(self) -> "Tensor":
        return dtorch.log2(self)

    def log10(self) -> "Tensor":
        return dtorch.log10(self)

    def isinf(self) -> "Tensor":
        return dtorch.isinf(self)

    def isnan(self) -> "Tensor":
        return dtorch.isnan(self)

    def square(self) -> "Tensor":
        return dtorch.square(self)

    def rsqrt(self) -> "Tensor":
        return dtorch.rsqrt(self)

    def abs(self) -> "Tensor":
        return dtorch.abs(self)

    def round(self) -> "Tensor":
        return dtorch.round(self)

    def floor(self) -> "Tensor":
        return dtorch.floor(self)

    def cos(self) -> "Tensor":
        return dtorch.cos(self)

    def sin(self) -> "Tensor":
        return dtorch.sin(self)

    def asin(self) -> "Tensor":
        return dtorch.asin(self)

    def tanh(self) -> "Tensor":
        return dtorch.nn.functional.tanh(self)

    def relu(self) -> "Tensor":
        return dtorch.nn.functional.relu(self)

    def sigmoid(self) -> "Tensor":
        return dtorch.nn.functional.sigmoid(self)

    def squeeze(self, dim: Optional[Union[int, List[int]]]) -> "Tensor":
        return dtorch.squeeze(self, dim)

    def softmax(self, dim: int) -> "Tensor":
        return dtorch.softmax(self, dim)

    def expand(self, *args) -> "Tensor":
        args = args[0] if len(args) == 1 and (isinstance(args[0], list) or isinstance(args[0], tuple)) else args
        shape = torch.Size(args)
        return dtorch.nn.functional._expand(self, shape)

    def view(self, *args) -> "Tensor":
        args = args[0] if len(args) == 1 else args
        if isinstance(args, Sequence) and len(args) > 0 and isinstance(args[0], Placement):
            # view placements
            return dtorch.nn.functional._view(self, args)
        else:
            # view shape
            shape = torch.Size(args)
            return dtorch.nn.functional._view(self, shape)

    def is_contiguous(self) -> "Tensor":
        # TODO: support tensor.is_contiguous()
        return True

    def contiguous(self) -> "Tensor":
        return dtorch.nn.functional._contiguous(self)

    def unsqueeze(self, dim: int) -> "Tensor":
        return dtorch.unsqueeze(self, dim)

    def permute(self, *args) -> "Tensor":
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            dims = args[0]
        else:
            dims = args
        return dtorch.permute(self, dims)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        return dtorch.transpose(self, dim0, dim1)

    def repeat(self, *args, **kwargs) -> "Tensor":
        if len(args) == 1:
            if isinstance(args[0], int):
                repeats = list(args)
            elif isinstance(args[0], torch.Size):
                repeats = args[0]
            else:
                repeats = list(args[0])
        elif len(args) > 1:
            repeats = list(args)
        else:
            assert len(kwargs) == 1 and "repeats" in kwargs
            repeats = kwargs["repeats"]
        return dtorch.nn.functional._repeat(self, repeats)

    def chunk(self, chunks: int, dim: int = 0) -> "Tensor":
        return dtorch.chunk(self, chunks, dim)

    def unbind(self, dim: int = 0) -> "Tensor":
        return dtorch.unbind(self, dim)

    def flatten(self, start_dim=0, end_dim=-1) -> "Tensor":
        return dtorch.flatten(self, start_dim=start_dim, end_dim=end_dim)

    def eq(self, other) -> "Tensor":
        return dtorch.eq(self, other)

    def equal(self, other: "Tensor") -> bool:
        return dtorch.equal(self, other)

    def greater(self, other: "Tensor") -> "Tensor":
        return dtorch.greater(self, other)

    def less(self, other: "Tensor") -> "Tensor":
        return dtorch.less(self, other)

    def greater_equal(self, other: "Tensor") -> "Tensor":
        return dtorch.greater_equal(self, other)

    def less_equal(self, other: "Tensor") -> "Tensor":
        return dtorch.less_equal(self, other)

    def logical_and(self, other: "Tensor") -> "Tensor":
        return dtorch.logical_and(self, other)

    def logical_or(self, other: "Tensor") -> "Tensor":
        return dtorch.logical_or(self, other)

    def nonzero(self) -> "Tensor":
        return dtorch.nonzero(self)

    def max(self, *args, **kwargs) -> "Tensor":
        return dtorch.max(self, *args, **kwargs)

    def min(self, *args, **kwargs) -> "Tensor":
        return dtorch.min(self, *args, **kwargs)

    def argmax(self, *args, **kwargs) -> "Tensor":
        return dtorch.argmax(self, *args, **kwargs)

    def argmin(self, *args, **kwargs) -> "Tensor":
        return dtorch.argmin(self, *args, **kwargs)

    def maximum(self, other: "Tensor") -> "Tensor":
        return dtorch.maximum(self, other)

    def minimum(self, other: "Tensor") -> "Tensor":
        return dtorch.minimum(self, other)

    def sum(self, *args, **kwargs) -> "Tensor":
        return dtorch.sum(self, *args, **kwargs)

    def mean(self, *args, **kwargs) -> "Tensor":
        return dtorch.mean(self, *args, **kwargs)

    def any(self, *args, **kwargs) -> "Tensor":
        return dtorch.any(self, *args, **kwargs)

    def all(self, *args, **kwargs) -> "Tensor":
        return dtorch.all(self, *args, **kwargs)

    def outer(self, *args, **kwargs) -> "Tensor":
        return dtorch.outer(self, *args, **kwargs)

    def repeat_interleave(self, *args, **kwargs) -> "Tensor":
        return dtorch.repeat_interleave(self, *args, **kwargs)

    def clamp(self, min: Union[float, "Tensor"], max: Union[float, "Tensor"]) -> "Tensor":
        return dtorch.clamp(self, min, max)

    def clip(self, min: Union[float, "Tensor"], max: Union[float, "Tensor"]) -> "Tensor":
        return dtorch.clip(self, min, max)

    def reshape(self, *args, **kwargs) -> "Tensor":
        shape = None
        if len(args) > 0:
            assert len(kwargs) == 0
            shape = args
        else:
            assert "shape" in kwargs
            shape = kwargs["shape"]

        return dtorch.reshape(self, shape)

    def where(self, condition: "Tensor", other: Union[float, "Tensor"]) -> "Tensor":
        return dtorch.where(condition, self, other)

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        return dtorch.nn.functional._masked_fill(self, mask, value)

    def masked_fill_(self, mask: "Tensor", value: float) -> "Tensor":
        result = dtorch.nn.functional._masked_fill(self, mask, value)
        self._inplace_assignment(result)
        return self

    def masked_scatter(self, mask: "Tensor", source: "Tensor") -> "Tensor":
        return dtorch.nn.functional._masked_scatter(self, mask, source)

    def masked_scatter_(self, mask: "Tensor", source: "Tensor") -> "Tensor":
        result = dtorch.nn.functional._masked_scatter(self, mask, source)
        self._inplace_assignment(result)
        return self

    def __iter__(self):
        if self.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self):
            result = self[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    @staticmethod
    def _parse_index(indexs):
        is_int_seq = lambda obj: isinstance(obj, (list, tuple)) and all(isinstance(x, int) for x in obj)
        # tuple → multi-dimensional indexing, iterate over elements
        # list  → single tensor index (integer array indexing, aligns with PyTorch)
        if isinstance(indexs, tuple):
            dtorch_indexs = []
            for index in indexs:
                assert (
                    isinstance(index, int)
                    or isinstance(index, slice)
                    or isinstance(index, EllipsisType)
                    or index is None
                    or isinstance(index, dtorch.Tensor)
                    or is_int_seq(index)
                )
                # Convert to torch Tensor: 1. isinstance(index, dtorch.Tensor) and 2. is_int_seq(index)
                dtorch_indexs.append(dtorch._dtorch_py_api.Index(index))
            return dtorch_indexs
        else:
            assert (
                isinstance(indexs, int)
                or isinstance(indexs, slice)
                or isinstance(indexs, EllipsisType)
                or indexs is None
                or isinstance(indexs, dtorch.Tensor)
                or is_int_seq(indexs)
            )
            return (dtorch._dtorch_py_api.Index(indexs),)

    def __getitem__(self, indexs):
        dtorch_indexs = self._parse_index(indexs)
        return dtorch.nn.functional._get_item(self, dtorch_indexs)

    def __setitem__(self, indexs, value):
        if not isinstance(value, dtorch.Tensor):
            value = dtorch.tensor(value, device_mesh=self.device_mesh)

        dtorch_indexs = self._parse_index(indexs)
        return dtorch.nn.functional._set_item(self, value, dtorch_indexs)

    def __eq__(self, other):
        if isinstance(other, Tensor) or isinstance(other, numbers.Number):
            return self.eq(other)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Tensor) or isinstance(other, numbers.Number):
            return self.greater(other)
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Tensor) or isinstance(other, numbers.Number):
            return self.less(other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Tensor) or isinstance(other, numbers.Number):
            return self.greater_equal(other)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Tensor) or isinstance(other, numbers.Number):
            return self.less_equal(other)
        return NotImplemented


# To avoid confusion with torch.Tensor when there is an error in function argument passing, since both print "Tensor".
Tensor = DTorchTensor

# Operator dunders call dtorch.* directly instead of self.add/sub/... to skip the intermediate
# Python method call.  __rsub__ / __rtruediv__ also avoid an intermediate Tensor when `other`
# is a Tensor.  When `other` is a number we fall back to the two-step path because there is no
# _Sub(Scalar, Tensor) / _Div(Scalar, Tensor) overload.
Tensor.__add__ = lambda self, other: dtorch.add(self, other)
Tensor.__radd__ = lambda self, other: dtorch.add(self, other)
Tensor.__sub__ = lambda self, other: dtorch.sub(self, other)
Tensor.__rsub__ = lambda self, other: (
    dtorch.sub(other, self) if isinstance(other, Tensor) else dtorch.add(dtorch.neg(self), other)
)
Tensor.__mul__ = lambda self, other: dtorch.mul(self, other)
Tensor.__rmul__ = lambda self, other: dtorch.mul(self, other)
Tensor.__truediv__ = lambda self, other: dtorch.div(self, other)
Tensor.__rtruediv__ = lambda self, other: (
    dtorch.div(other, self) if isinstance(other, Tensor) else dtorch.mul(self.reciprocal(), other)
)
Tensor.__neg__ = lambda self: dtorch.neg(self)
Tensor.__pow__ = lambda self, other: dtorch.pow(self, other)
Tensor.__rpow__ = lambda self, other: dtorch.pow(other, self)


# 1. https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor
# 2. When the device and device_mesh of dtorch.tensor() are None and data isn't torch.Tensor, a tensor will be created
#    on the graph.default_device_mesh.
def tensor(
    data,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    graph: Optional["Graph"] = None,
) -> Tensor:
    assert not (device is not None and device_mesh is not None), "Can't set device and device_mesh at same time"

    if isinstance(data, torch.Tensor):
        torch_tensor = data
        if dtype is not None and dtype != torch_tensor.dtype:
            torch_tensor = torch_tensor.to(dtype=dtype)
    else:
        device_mesh = get_default_device_mesh(device=device, device_mesh=device_mesh, graph=graph)
        create_device = device_mesh.device_type
        torch_tensor = torch.tensor(data, dtype=dtype, device=create_device)

    if dtype is not None:
        assert dtype == torch_tensor.dtype

    return Tensor(
        torch_tensor,
        device=device,
        device_mesh=device_mesh,
        placements=placements,
        graph=graph,
    )


def tensors_redistribute_by_dict(
    tensors: Sequence[Tensor],
    *args,
    **kwargs,
):
    return [
        tensor.redistribute_by_dict(
            *args,
            **kwargs,
        )
        for tensor in tensors
    ]


class TensorFuture:
    """Python wrapper around C++ TensorFuture for async tensor value retrieval.

    Returned by DTorchTensor.to_torch_async(). The user can call get() to block
    until the tensor value is ready, or check is_ready() to poll without blocking.
    """

    def __init__(self, cpp_future):
        self._cpp_future = cpp_future

    def get(self) -> torch.Tensor:
        """Block until the tensor value is ready and return it."""
        return self._cpp_future.Get()

    def wait(self) -> None:
        """Block until the tensor value is ready."""
        self._cpp_future.Wait()

    def wait_for(self, timeout_ms: int) -> bool:
        """Wait up to timeout_ms milliseconds. Returns True if ready, False on timeout."""
        return self._cpp_future.WaitFor(timeout_ms)

    def is_ready(self) -> bool:
        """Return True if the tensor value is ready (non-blocking)."""
        return self._cpp_future.IsReady()

    def __await__(self):
        delay = 0.0005
        while not self.is_ready():
            yield from asyncio.sleep(delay).__await__()
        return self.get()
