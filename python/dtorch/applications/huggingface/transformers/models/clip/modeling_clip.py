# Copyright 2020 The HuggingFace Team. All rights reserved.
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


from dataclasses import dataclass
import math
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

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig
from transformers.utils import ModelOutput


@dataclass
class CLIPTextModelOutput(ModelOutput):
    text_embeds: Optional[dtorch.FloatTensor] = None
    last_hidden_state: dtorch.FloatTensor = None
    hidden_states: Optional[Tuple[dtorch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[dtorch.FloatTensor, ...]] = None


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.EmbeddingWithReplicateOutput(config.vocab_size, embed_dim)
        self.position_embedding = nn.EmbeddingWithReplicateOutput(config.max_position_embeddings, embed_dim)

        self.register_buffer(
            "position_ids",
            dtorch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[dtorch.Tensor] = None,
        inputs_embeds: Optional[dtorch.Tensor] = None,
    ) -> dtorch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPSdpaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.ColumnParallelLinear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.ColumnParallelLinear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.ColumnParallelLinear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.RowParallelLinearWithReplicateOutput(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: dtorch.Tensor,
        attention_mask: Optional[dtorch.Tensor] = None,
        causal_attention_mask: Optional[dtorch.Tensor] = None,
    ) -> Tuple[dtorch.Tensor, Optional[dtorch.Tensor]]:
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, query_states.shape[1], -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, key_states.shape[1], -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, value_states.shape[1], -1, self.head_dim).transpose(1, 2)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask

        # Align with PyTorch's sdpa_attention_forward: is_causal is True only when
        # query.shape[2] > 1 (seq_len > 1) and no attention_mask is provided.
        is_causal = query_states.shape[2] > 1 and attention_mask is None and self.is_causal
        dropout_p = 0.0 if not self.training else self.dropout

        attn_output = dtorch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=self.scale,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.ColumnParallelLinear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.RowParallelLinearWithReplicateOutput(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: dtorch.Tensor) -> dtorch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPSdpaAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: dtorch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[dtorch.Tensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class CLIPPreTrainedModel(PreTrainedModel):
    config_class = CLIPConfig


class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = [] if output_hidden_states else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + [
                    hidden_states,
                ]
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
            )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            encoder_states = encoder_states + [
                hidden_states,
            ]

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)


class CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.eos_token_id = config.eos_token_id

    def forward(
        self,
        input_ids: Optional[dtorch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pooled_output_index: Optional[List[int]] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device_mesh=hidden_states.device_mesh
        )
        # Align mask placements with input_ids so the batch dim is sharded consistently
        # (e.g. dp=2: mask [2,1,64,64] -> local [1,1,64,64], matching query's local batch).
        causal_attention_mask = causal_attention_mask.redistribute_like(input_ids)

        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            last_hidden_state = last_hidden_state.redistribute(
                placements=[Replicate()] * len(last_hidden_state.placements)
            )
            if pooled_output_index is not None:
                # Avoid using a tensor as an index, as it breaks the computation graph.
                pooled_output = last_hidden_state[
                    :,
                    pooled_output_index,
                ]
                pooled_output = pooled_output.squeeze(dim=1)
            else:
                pooled_output = last_hidden_state[
                    dtorch.arange(last_hidden_state.shape[0], device_mesh=last_hidden_state.device_mesh),
                    input_ids.to(dtype=dtorch.int, device_mesh=last_hidden_state.device_mesh).argmax(dim=-1),
                ]
        else:
            # pooled_output = last_hidden_state[
            #     dtorch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            #     (
            #         input_ids.to(dtype=torch.int32, device=last_hidden_state.device)
            #         == self.eos_token_id
            #     )
            #     .int()
            #     .argmax(dim=-1),
            # ]
            pooled_output = last_hidden_state[:, 0]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    def __init__(
        self,
        config: CLIPTextConfig,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        super().__init__(config)

        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        device_mesh.check_all_dim_names_in_set({"dp", "tp"})

        with Graph.default_graph().device_mesh_guard(device_mesh):
            self.text_model = CLIPTextTransformer(config)

    def redistribute_input(
        self,
        input_ids: dtorch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pooled_output_index: Optional[List[int]] = None,
    ):
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
        return [input_ids, attention_mask, output_hidden_states, return_dict, pooled_output_index], {}

    def redistribute_output(self, output):
        if output.last_hidden_state is not None:
            output.last_hidden_state = output.last_hidden_state.redistribute(
                device_mesh=self.input_device_mesh, placements=self.input_placement
            )

        if output.pooler_output is not None:
            output.pooler_output = output.pooler_output.redistribute(
                device_mesh=self.input_device_mesh, placements=self.input_placement
            )

        return output

    def forward(
        self,
        input_ids: Optional[dtorch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pooled_output_index: Optional[List[int]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pooled_output_index=pooled_output_index,
        )


class CLIPTextModelWithProjection(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    def __init__(
        self,
        config: CLIPTextConfig,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        super().__init__(config)
        self.config = config

        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        device_mesh.check_all_dim_names_in_set({"dp", "tp"})

        with Graph.default_graph().device_mesh_guard(device_mesh):
            self.text_model = CLIPTextTransformer(config)
            self.text_projection = nn.ColumnParallelLinear(config.hidden_size, config.projection_dim, bias=False)

    def redistribute_input(
        self,
        input_ids: dtorch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pooled_output_index: Optional[List[int]] = None,
    ):
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
        return [input_ids, attention_mask, output_hidden_states, return_dict, pooled_output_index], {}

    def redistribute_output(self, output):
        output.text_embeds = output.text_embeds.redistribute(
            device_mesh=self.input_device_mesh, placements=self.input_placement
        )

        if output.last_hidden_state is not None:
            output.last_hidden_state = output.last_hidden_state.redistribute(
                device_mesh=self.input_device_mesh, placements=self.input_placement
            )

        # TODO: tmp fix, sd3 use this tensor
        output.hidden_states[-2] = output.hidden_states[-2].redistribute(
            device_mesh=self.input_device_mesh, placements=self.input_placement
        )

        return output

    def forward(
        self,
        input_ids: Optional[dtorch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pooled_output_index: Optional[List[int]] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pooled_output_index=pooled_output_index,
        )

        pooled_output = text_outputs[1]

        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )
