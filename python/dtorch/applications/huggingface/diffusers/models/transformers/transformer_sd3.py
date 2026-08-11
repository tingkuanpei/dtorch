# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import math
from typing import List, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config

import dtorch
import dtorch.nn as nn
import dtorch.nn.functional as F
from dtorch import (
    DeviceMesh,
    Graph,
    Shard,
    Replicate,
    get_default_device_mesh,
)
from dtorch.applications.core.config import ExecuteConfig

from ...models.modeling_utils import ModelMixin
from ...models.attention import FeedForward, JointTransformerBlock
from ..embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from ...models.normalization import AdaLayerNormContinuous, AdaLayerNormZero


class SD3Transformer2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[int, ...] = (),
        qk_norm: Optional[str] = None,
        device_mesh: Optional[DeviceMesh] = None,
        execute_config: ExecuteConfig = ExecuteConfig(),
    ):
        super().__init__()

        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        device_mesh.check_all_dim_names_in_set({"dp", "tp", "ulysess_cp", "ring_cp"})

        with Graph.default_graph().device_mesh_guard(device_mesh):
            default_out_channels = in_channels
            self.out_channels = out_channels if out_channels is not None else default_out_channels
            self.inner_dim = num_attention_heads * attention_head_dim
            self.patch_size = patch_size

            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=self.inner_dim,
                pos_embed_max_size=pos_embed_max_size,
            )
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=pooled_projection_dim,
            )
            self.context_embedder = nn.ColumnParallelLinearWithReplicateOutput(
                joint_attention_dim, caption_projection_dim
            )

            self.transformer_blocks = nn.ModuleList(
                [
                    JointTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        context_pre_only=i == num_layers - 1,
                        qk_norm=qk_norm,
                        use_dual_attention=True if i in dual_attention_layers else False,
                        execute_config=execute_config,
                    )
                    for i in range(num_layers)
                ]
            )

            self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out = nn.ColumnParallelLinearWithReplicateOutput(
                self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
            )

    def redistribute_input(
        self,
        hidden_states: dtorch.Tensor,
        encoder_hidden_states: dtorch.Tensor = None,
        pooled_projections: dtorch.Tensor = None,
        timestep: dtorch.Tensor = None,
    ):
        device_mesh = self.first_param_device_mesh()

        placements_dict = {
            "dp": Shard(0),
            "tp": Replicate(),
            "ulysess_cp": Shard(2),
            "ring_cp": Shard(2),
        }

        hidden_states = hidden_states.redistribute_by_dict(
            device_mesh=device_mesh,
            placements_dict=placements_dict,
        )

        placements_dict["ulysess_cp"] = Shard(1)
        placements_dict["ring_cp"] = Shard(1)
        encoder_hidden_states = encoder_hidden_states.redistribute_by_dict(
            device_mesh=device_mesh,
            placements_dict=placements_dict,
        )

        placements_dict["ulysess_cp"] = Replicate()
        placements_dict["ring_cp"] = Replicate()
        pooled_projections = pooled_projections.redistribute_by_dict(
            device_mesh=device_mesh,
            placements_dict=placements_dict,
        )

        timestep = timestep.redistribute_by_dict(
            device_mesh=device_mesh,
            placements_dict=placements_dict,
        )

        return [hidden_states, encoder_hidden_states, pooled_projections, timestep], {}

    def forward(
        self,
        hidden_states: dtorch.Tensor,
        encoder_hidden_states: dtorch.Tensor = None,
        pooled_projections: dtorch.Tensor = None,
        timestep: dtorch.Tensor = None,
    ):
        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = dtorch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        return (output,)
