"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import List, Optional, Union, Tuple

import torch
import torch.nn.functional as F


@torch.compile()
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_real:
        # cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


@torch.compile()
def silu_linear_chunk(
    emb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    chunk_size: int,
):
    emb = F.linear(F.silu(emb), linear_weight, linear_bias)
    return emb.chunk(chunk_size, dim=1)


@torch.compile()
def layer_norm_mul_add(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    normalized_shape: List[int],
    norm_eps: float,
    norm_scale: Optional[torch.Tensor],
    norm_bias: Optional[torch.Tensor],
):
    x = F.layer_norm(x, normalized_shape, norm_scale, norm_bias, norm_eps)
    x = x * (1 + scale[:, None]) + shift[:, None]
    return x


@torch.compile()
def rms_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = None,
):
    return torch.nn.functional.rms_norm(input, normalized_shape, weight, eps)
