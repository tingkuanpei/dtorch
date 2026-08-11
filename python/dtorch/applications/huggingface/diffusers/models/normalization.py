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

import numbers
from typing import Optional, Tuple

import dtorch
import dtorch.nn.functional as F
from dtorch import nn


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.ReplicateParallelLinear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            assert 0 == 1
            # TODO: convert input as float32
            # self.norm = nn.RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: dtorch.Tensor, conditioning_embedding: dtorch.Tensor) -> dtorch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = dtorch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


LayerNorm = nn.LayerNorm


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: dtorch.Tensor) -> dtorch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class RMSNorm(nn.Module):
    r"""
    RMS Norm as introduced in https://huggingface.co/papers/1910.07467 by Zhang et al.

    Args:
        dim (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
        eps (`float`): Small value to use when calculating the reciprocal of the square-root.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        bias (`bool`, defaults to False): If also training the `bias` param.
    """

    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = dtorch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(dtorch.ones(dim))
            if bias:
                self.bias = nn.Parameter(dtorch.zeros(dim))

    def forward(self, hidden_states):
        # if is_torch_npu_available():
        if False:
            import torch_npu

            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [dtorch.float16, dtorch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(dtorch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * dtorch.rsqrt(variance + self.eps)

            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [dtorch.float16, dtorch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                hidden_states = hidden_states * self.weight
                if self.bias is not None:
                    hidden_states = hidden_states + self.bias
            else:
                hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class AdaLayerNormZero(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        norm_type="layer_norm",
        bias=True,
    ):
        super().__init__()
        assert num_embeddings is None
        self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.ReplicateParallelLinear(embedding_dim, 6 * embedding_dim, bias=bias)
        self.norm_type = norm_type
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: dtorch.Tensor,
        timestep: Optional[dtorch.Tensor] = None,
        class_labels: Optional[dtorch.Tensor] = None,
        hidden_dtype: Optional[dtorch.dtype] = None,
        emb: Optional[dtorch.Tensor] = None,
        fuse_silu_linear_chunk=False,
        fuse_layer_norm_mul_add=False,
    ) -> Tuple[dtorch.Tensor, dtorch.Tensor, dtorch.Tensor, dtorch.Tensor, dtorch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)

        if fuse_silu_linear_chunk:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = dtorch.nn.functional._silu_linear_chunk(emb, self.linear.weight, self.linear.bias, 6)
        else:
            emb = self.linear(self.silu(emb))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)

        if fuse_layer_norm_mul_add:
            assert self.norm_type == "layer_norm"
            x = dtorch.nn.functional._layer_norm_mul_add(
                x,
                scale_msa,
                shift_msa,
                self.norm.normalized_shape,
                self.norm.eps,
                self.norm.weight,
                self.norm.bias,
            )
        else:
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.ReplicateParallelLinear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: dtorch.Tensor,
        emb: Optional[dtorch.Tensor] = None,
        fuse_silu_linear_chunk=False,
        fuse_layer_norm_mul_add=False,
    ) -> Tuple[dtorch.Tensor, dtorch.Tensor, dtorch.Tensor, dtorch.Tensor, dtorch.Tensor]:
        if fuse_silu_linear_chunk:
            (
                shift_msa,
                scale_msa,
                gate_msa,
            ) = dtorch.nn.functional._silu_linear_chunk(emb, self.linear.weight, self.linear.bias, 3)
        else:
            emb = self.linear(self.silu(emb))
            shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)

        if fuse_layer_norm_mul_add:
            x = dtorch.nn.functional._layer_norm_mul_add(
                x,
                scale_msa,
                shift_msa,
                self.norm.normalized_shape,
                self.norm.eps,
                self.norm.weight,
                self.norm.bias,
            )
        else:
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa
