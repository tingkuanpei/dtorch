"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Union, Optional, Sequence
import functools

import torch

import dtorch
from dtorch import (
    Tensor,
    DeviceMesh,
    Placement,
    Replicate,
    Shard,
)
from dtorch.distributed_spec import (
    get_default_device_mesh,
)
from dtorch.nn.parameter import Parameter
from .module import Module
from .. import functional as F


class Embedding(Module):
    num_embeddings: int
    embedding_dim: int
    weight: Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        device_mesh: Optional[DeviceMesh] = None,
        tp_dim: Optional[Union[int, str]] = "tp",
    ) -> None:
        super(Embedding, self).__init__()
        assert isinstance(tp_dim, str) or isinstance(tp_dim, int)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_dim = tp_dim

        self.device_mesh = get_default_device_mesh(device, device_mesh)
        placements = [Replicate()] * self.device_mesh.ndim

        # tp_dim specifies the dimension for tensor parallelism (TP) sharding, which can be either str or int.
        # Notes:
        # 1. If tp_dim is a string, TP sharding will not be performed if the match dimention names fails.
        # 2. If the number of shards is 1, TP sharding will not be performed.
        if isinstance(self.tp_dim, str):
            self.tp_dim = self.device_mesh.dim_name_index(self.tp_dim)

        if self.tp_dim is not None and self.device_mesh.shape[self.tp_dim] == 1:
            self.tp_dim = None

        if self.tp_dim is not None:
            placements[self.tp_dim] = Shard(1)

        factory_kwargs = {
            "dtype": dtype,
            "device_mesh": get_default_device_mesh(device, device_mesh),
            "placements": placements,
        }
        self.weight = Parameter(dtorch.empty(num_embeddings, embedding_dim, **factory_kwargs))

    def redistribute_input(self, input: Tensor):
        if self.tp_dim is not None:
            assert input.placements[self.tp_dim] == Replicate()

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
        return F.embedding(input, self.weight)


class EmbeddingWithReplicateOutput(Embedding):
    def redistribute_output(self, output: Tensor):
        return super(EmbeddingWithReplicateOutput, self).redistribute_output(output, Replicate())
