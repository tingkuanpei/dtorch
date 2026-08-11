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


from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput

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
from dtorch.tensor import tensors_redistribute_by_dict

from ...models.modeling_utils import ModelMixin
from ...models.attention import FeedForward
from ...models.attention_processor import (
    Attention,
    FluxAttnProcessor2_0,
)
from ...models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from ..embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    FluxPosEmbed,
)


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        mlp_ratio=4.0,
        execute_config: ExecuteConfig = ExecuteConfig(),
    ):
        super().__init__()

        fuse_kernel_config = execute_config.fuse_kernel_config
        self.fuse_silu_linear_chunk = fuse_kernel_config.fuse_silu_linear_chunk
        self.fuse_layer_norm_mul_add = fuse_kernel_config.fuse_layer_norm_mul_add

        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.ColumnParallelLinear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.ColumnParallelLinearWithReplicateInputOutput(dim + self.mlp_hidden_dim, dim)

        processor = FluxAttnProcessor2_0(
            execute_config=execute_config,
        )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: dtorch.FloatTensor,
        encoder_hidden_states: dtorch.Tensor,
        temb: dtorch.FloatTensor,
        image_rotary_emb=None,
    ):
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = dtorch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(
            hidden_states,
            emb=temb,
            fuse_silu_linear_chunk=self.fuse_silu_linear_chunk,
            fuse_layer_norm_mul_add=self.fuse_layer_norm_mul_add,
        )
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = dtorch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == dtorch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len].contiguous(),
            hidden_states[:, text_seq_len:].contiguous(),
        )
        return encoder_hidden_states, hidden_states


class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        qk_norm="rms_norm",
        eps=1e-6,
        execute_config: ExecuteConfig = ExecuteConfig(),
    ):
        super().__init__()

        fuse_kernel_config = execute_config.fuse_kernel_config
        self.fuse_silu_linear_chunk = fuse_kernel_config.fuse_silu_linear_chunk
        self.fuse_layer_norm_mul_add = fuse_kernel_config.fuse_layer_norm_mul_add

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0(
                execute_config=execute_config,
            )
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: dtorch.FloatTensor,
        encoder_hidden_states: dtorch.FloatTensor,
        temb: dtorch.FloatTensor,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states,
            emb=temb,
            fuse_silu_linear_chunk=self.fuse_silu_linear_chunk,
            fuse_layer_norm_mul_add=self.fuse_layer_norm_mul_add,
        )

        (norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp,) = self.norm1_context(
            encoder_hidden_states,
            emb=temb,
            fuse_silu_linear_chunk=self.fuse_silu_linear_chunk,
            fuse_layer_norm_mul_add=self.fuse_layer_norm_mul_add,
        )
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.fuse_layer_norm_mul_add:
            norm_hidden_states = dtorch.nn.functional._layer_norm_mul_add(
                hidden_states,
                scale_mlp,
                shift_mlp,
                self.norm2.normalized_shape,
                self.norm2.eps,
                self.norm2.weight,
                self.norm2.bias,
            )
        else:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        if self.fuse_layer_norm_mul_add:
            norm_encoder_hidden_states = dtorch.nn.functional._layer_norm_mul_add(
                encoder_hidden_states,
                c_scale_mlp,
                c_shift_mlp,
                self.norm2_context.normalized_shape,
                self.norm2_context.eps,
                self.norm2_context.weight,
                self.norm2_context.bias,
            )
        else:
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == dtorch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        device_mesh: Optional[DeviceMesh] = None,
        execute_config: ExecuteConfig = ExecuteConfig(),
    ):
        super().__init__()

        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        device_mesh.check_all_dim_names_in_set({"dp", "tp", "ulysess_cp", "ring_cp"})

        with Graph.default_graph().device_mesh_guard(device_mesh):
            self.out_channels = out_channels or in_channels
            self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

            self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

            text_time_guidance_cls = (
                CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
            )
            self.time_text_embed = text_time_guidance_cls(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=self.config.pooled_projection_dim,
            )

            self.context_embedder = nn.ColumnParallelLinearWithReplicateOutput(
                self.config.joint_attention_dim, self.inner_dim
            )
            self.x_embedder = nn.ColumnParallelLinearWithReplicateOutput(self.config.in_channels, self.inner_dim)

            self.transformer_blocks = nn.ModuleList(
                [
                    FluxTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        attention_head_dim=self.config.attention_head_dim,
                        execute_config=execute_config,
                    )
                    for i in range(self.config.num_layers)
                ]
            )

            self.single_transformer_blocks = nn.ModuleList(
                [
                    FluxSingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        attention_head_dim=self.config.attention_head_dim,
                        execute_config=execute_config,
                    )
                    for i in range(self.config.num_single_layers)
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
        timestep: dtorch.LongTensor = None,
        img_ids: dtorch.Tensor = None,
        txt_ids: dtorch.Tensor = None,
        guidance: dtorch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[dtorch.FloatTensor, Transformer2DModelOutput]:
        device_mesh = self.first_param_device_mesh()

        placements_dict = {
            "dp": Shard(0),
            "tp": Replicate(),
            "ulysess_cp": Shard(1),
            "ring_cp": Shard(1),
        }

        hidden_states, encoder_hidden_states = tensors_redistribute_by_dict(
            [hidden_states, encoder_hidden_states],
            device_mesh=device_mesh,
            placements_dict=placements_dict,
        )

        placements_dict["ulysess_cp"] = Shard(0)
        placements_dict["ring_cp"] = Shard(0)
        img_ids, txt_ids = tensors_redistribute_by_dict(
            [img_ids, txt_ids],
            device_mesh=device_mesh,
            placements_dict=placements_dict,
        )

        placements_dict["ulysess_cp"] = Replicate()
        placements_dict["ring_cp"] = Replicate()
        pooled_projections, timestep, guidance = tensors_redistribute_by_dict(
            [pooled_projections, timestep, guidance],
            device_mesh=device_mesh,
            placements_dict=placements_dict,
        )

        return [
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
            return_dict,
        ], {}

    def forward(
        self,
        hidden_states: dtorch.Tensor,
        encoder_hidden_states: dtorch.Tensor = None,
        pooled_projections: dtorch.Tensor = None,
        timestep: dtorch.LongTensor = None,
        img_ids: dtorch.Tensor = None,
        txt_ids: dtorch.Tensor = None,
        guidance: dtorch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[dtorch.FloatTensor, Transformer2DModelOutput]:
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        assert txt_ids.ndim == 2
        assert img_ids.ndim == 2
        ids = dtorch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
