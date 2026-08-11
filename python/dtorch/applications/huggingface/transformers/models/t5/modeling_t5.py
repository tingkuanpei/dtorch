"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import math
import copy
from typing import List, Optional, Tuple, Union

import torch

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
from dtorch.nn.parameter import Parameter

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from transformers.models.t5.configuration_t5 import T5Config


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(dtorch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * dtorch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# class T5LayerNorm(nn.RMSNorm):
#     def forward(self, x: dtorch.Tensor) -> dtorch.Tensor:
#         origin_dtype = x.dtype
#         return F.rms_norm(
#             x.float(),
#             self.normalized_shape,
#             self.weight.float() if self.weight is not None else None,
#             self.eps,
#         ).to(origin_dtype)


class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        assert not config.is_decoder
        self.is_decoder = False
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx

        self.q = nn.ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.RowParallelLinearWithReplicateOutput(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(dtorch.long) * num_buckets
            relative_position = dtorch.abs(relative_position)
        else:
            relative_position = -dtorch.min(relative_position, dtorch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            dtorch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(dtorch.long)
        relative_position_if_large = dtorch.min(
            relative_position_if_large,
            dtorch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += dtorch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device_mesh=None, cache_position=None):
        if device_mesh is None:
            device_mesh = self.relative_attention_bias.weight.device_mesh
        if cache_position is None:
            context_position = dtorch.arange(query_length, dtype=dtorch.long, device_mesh=device_mesh)[:, None]
        else:
            context_position = cache_position[:, None].to(device_mesh=device_mesh)
        memory_position = dtorch.arange(key_length, dtype=dtorch.long, device_mesh=device_mesh)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states,
        position_bias=None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        current_states = hidden_states
        key_states = self.k(current_states)
        value_states = self.v(current_states)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if position_bias is None:
            key_length = key_states.shape[-2]
            real_seq_length = query_states.shape[-2]
            if not self.has_relative_attention_bias:
                position_bias = dtorch.zeros(
                    (1, self.n_heads, seq_length, key_length),
                    device_mesh=query_states.device_mesh,
                    dtype=query_states.dtype,
                )
            else:
                position_bias = self.compute_bias(
                    real_seq_length,
                    key_length,
                    device_mesh=query_states.device_mesh,
                )
                position_bias = position_bias[:, :, -seq_length:, :]

        # Using scaled_dot_product_attention in the SD3 model leads to significant differences in the output images.
        # attn_output = dtorch.nn.functional.scaled_dot_product_attention(
        #     query_states, key_states, value_states, attn_mask=position_bias, scale=1.0
        # )
        scores = dtorch.matmul(query_states, key_states.transpose(3, 2))
        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = dtorch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            layer_idx=layer_idx,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.RowParallelLinearWithReplicateOutput(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, dtorch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != dtorch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.RowParallelLinearWithReplicateOutput(config.d_ff, config.d_model, bias=False, dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout_rate)
        assert config.dense_act_fn == "relu"
        self.act = nn.ReLU()

    # Temp support _keep_in_fp32_modules = ["wo"]
    def _apply(self, fn, recurse=True):
        wo_weight = self.wo.weight
        super()._apply(fn, recurse)
        self.wo.weight = Parameter(wo_weight.to(device_mesh=self.wo.weight.device_mesh))
        return self

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, dtorch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != dtorch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                layer_idx=layer_idx,
            )
        )
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        position_bias=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            position_bias=position_bias,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]

        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == dtorch.float16:
            clamp_value = dtorch.where(
                dtorch.isinf(hidden_states).any(),
                dtorch.finfo(hidden_states.dtype).max - 1000,
                dtorch.finfo(hidden_states.dtype).max,
            )
            hidden_states = dtorch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        outputs = outputs + attention_outputs
        return outputs


class T5PreTrainedModel(PreTrainedModel):
    config_class = T5Config
    _keep_in_fp32_modules = ["wo"]


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids=None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = input_shape
        position_bias = None
        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_outputs = layer_module(
                hidden_states,
                position_bias=position_bias,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return [
            hidden_states,
        ]


class T5EncoderModel(T5PreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    def __init__(
        self,
        config: T5Config,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        super().__init__(config)
        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        device_mesh.check_all_dim_names_in_set({"dp", "tp"})

        with Graph.default_graph().device_mesh_guard(device_mesh):
            self.shared = nn.EmbeddingWithReplicateOutput(config.vocab_size, config.d_model)

            encoder_config = copy.deepcopy(config)
            encoder_config.use_cache = False
            encoder_config.is_encoder_decoder = False
            self.encoder = T5Stack(encoder_config, self.shared)

    def forward(
        self,
        input_ids: Optional[dtorch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        assert not output_hidden_states

        input_device_mesh, input_placement = input_ids.device_mesh, input_ids.placements

        input_ids = input_ids.redistribute_by_dict(
            self.first_param_device_mesh(),
            placements_dict={
                "dp": Shard(0),
                "tp": Replicate(),
            },
        )

        encoder_outputs = self.encoder(
            input_ids=input_ids,
        )

        encoder_outputs[0] = encoder_outputs[0].redistribute(input_device_mesh, placements=input_placement)

        return encoder_outputs
