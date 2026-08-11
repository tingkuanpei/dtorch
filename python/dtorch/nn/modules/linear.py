"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Optional, Union, Sequence, Literal
import functools

import torch

import dtorch
from dtorch import (
    Tensor,
    DeviceMesh,
    Placement,
    Replicate,
    Partial,
    Shard,
)
from dtorch.distributed_spec import (
    get_default_device_mesh,
)
from dtorch.nn.parameter import Parameter
from .module import Module
from .. import functional as F


RowOrCol = Literal["row", "col"]


class Linear(Module):
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        device_mesh: Optional[DeviceMesh] = None,
        *,
        tp_dim: Optional[Union[int, str]] = "tp",
        tp_shard_type: Optional[RowOrCol] = None,
    ) -> None:
        super(Linear, self).__init__()
        if tp_dim is not None:
            assert isinstance(tp_dim, str) or isinstance(tp_dim, int)

        self.in_features = in_features
        self.out_features = out_features
        self.tp_dim = tp_dim
        self.tp_shard_type = tp_shard_type

        self.device_mesh = get_default_device_mesh(device, device_mesh)
        weight_placements = [Replicate()] * self.device_mesh.ndim
        bias_placements = [Replicate()] * self.device_mesh.ndim

        # tp_dim specifies the dimension for tensor parallelism (TP) sharding, which can be either str or int.
        # Notes:
        # 1. If tp_dim is a string, TP sharding will not be performed if the match dimention names fails.
        # 2. If the number of shards is 1, TP sharding will not be performed.
        if isinstance(self.tp_dim, str):
            self.tp_dim = self.device_mesh.dim_name_index(self.tp_dim)

        if self.tp_dim is not None:
            assert tp_shard_type == "row" or tp_shard_type == "col"

            weight_placements[self.tp_dim] = Shard(1) if tp_shard_type == "row" else Shard(0)
            bias_placements[self.tp_dim] = Partial() if tp_shard_type == "row" else Shard(0)

        factory_kwargs = {
            "dtype": dtype,
            "device_mesh": self.device_mesh,
            "placements": weight_placements,
        }
        self.weight = Parameter(dtorch.empty(out_features, in_features, **factory_kwargs))

        if bias:
            factory_kwargs["placements"] = bias_placements
            self.bias = Parameter(dtorch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def redistribute_input(self, input: Tensor, input_placement: Optional[Placement] = None):
        if self.tp_dim is not None and input_placement is not None:
            input = input.redistribute_by_dict(
                placements_dict={
                    self.tp_dim: input_placement,
                },
                default_placement_mode="keep",
            )

        if self.tp_dim is not None:
            expect_input_tp_placement = Shard(input.dim() - 1) if self.tp_shard_type == "row" else Replicate()
            assert input.check_placement(
                self.tp_dim, expect_input_tp_placement
            ), f"Invalid placement({input.placements}) in index {self.tp_dim}, expect: {expect_input_tp_placement}"

        return [input], {}

    def redistribute_output(self, output: Tensor, output_placement: Optional[Placement] = None):
        if self.tp_dim is not None and output_placement is not None:
            output = output.redistribute_by_dict(
                placements_dict={
                    self.tp_dim: output_placement,
                },
                default_placement_mode="keep",
            )
        return output

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class RowParallelLinear(Linear):
    def __init__(self, *args, **kwargs) -> None:
        super(RowParallelLinear, self).__init__(*args, **kwargs, tp_shard_type="row")


class RowParallelLinearWithReplicateOutput(RowParallelLinear):
    def redistribute_output(self, output: Tensor):
        return super(RowParallelLinearWithReplicateOutput, self).redistribute_output(output, Replicate())


class ColumnParallelLinear(Linear):
    def __init__(self, *args, **kwargs) -> None:
        super(ColumnParallelLinear, self).__init__(*args, **kwargs, tp_shard_type="col")


class ColumnParallelLinearWithReplicateOutput(ColumnParallelLinear):
    def redistribute_output(self, output: Tensor):
        return super(ColumnParallelLinearWithReplicateOutput, self).redistribute_output(output, Replicate())


class ColumnParallelLinearWithReplicateInputOutput(ColumnParallelLinear):
    def redistribute_input(self, input: Tensor):
        return super(ColumnParallelLinearWithReplicateInputOutput, self).redistribute_input(input, Replicate())

    def redistribute_output(self, output: Tensor):
        return super(ColumnParallelLinearWithReplicateInputOutput, self).redistribute_output(output, Replicate())


class ReplicateParallelLinear(Linear):
    def __init__(self, *args, **kwargs) -> None:
        super(ReplicateParallelLinear, self).__init__(*args, **kwargs, tp_dim=None)
