"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import math
from typing import List, Optional, Tuple, Union

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers import LlamaConfig
import torch

import dtorch
import dtorch.nn as nn
from dtorch.distributed_spec import (
    get_default_device_mesh,
)
from dtorch import Graph, DeviceMesh, Placement, Shard, Replicate, Tensor


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return dtorch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: dtorch.Tensor, n_rep: int) -> dtorch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaSdpaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True

        self.q_proj = nn.ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.RowParallelLinearWithReplicateOutput(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: dtorch.Tensor,
        position_embeddings,
    ) -> Tuple[dtorch.Tensor, Optional[dtorch.Tensor], Optional[Tuple[dtorch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = dtorch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaMLP(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias,
        )
        self.up_proj = nn.ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.RowParallelLinearWithReplicateOutput(
            self.intermediate_size,
            self.hidden_size,
            bias=config.mlp_bias,
        )

    def forward(self, x):
        down_proj = self.down_proj(dtorch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaSdpaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: dtorch.Tensor,
        position_embeddings: Tuple[dtorch.Tensor, dtorch.Tensor],
    ) -> dtorch.Tensor:
        # attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # mlp
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        # torch module, return torch.Tensor
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.embed_tokens = nn.EmbeddingWithReplicateOutput(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: dtorch.Tensor = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = [dtorch.Tensor(it, device_mesh=hidden_states.device_mesh) for it in position_embeddings]

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size

        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        device_mesh.check_all_dim_names_in_set({"dp", "tp"})

        with Graph.default_graph().device_mesh_guard(device_mesh):
            self.model = LlamaModel(config)
            self.lm_head = nn.ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

    def redistribute_input(self, input_ids: Tensor):
        self.input_device_mesh, self.input_placement = (
            input_ids.device_mesh,
            input_ids.placements,
        )

        input_ids = input_ids.redistribute_by_dict(
            self.first_param_device_mesh(),
            placements_dict={
                "dp": Shard(0),
                "tp": Replicate(),
            },
        )
        return [input_ids], {}

    def redistribute_output(self, logits: Tensor):
        return logits.redistribute(self.input_device_mesh, placements=self.input_placement)

    def forward(
        self,
        input_ids: dtorch.Tensor = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
        )
        logits = self.lm_head(hidden_states)

        return logits
