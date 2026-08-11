"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Optional

import dtorch
from dtorch import Tensor, Replicate, Shard, SdpaOption
import dtorch.nn.functional as F


def scaled_dot_product_attention_with_cp(
    query: Tensor,  # N, Hq, L, E
    key: Tensor,  # N, H, S, E
    value: Tensor,  # N, H, S, E
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa=False,
    sdpa_option: SdpaOption = SdpaOption(),
):
    assert sdpa_option is not None
    assert attn_mask is None
    assert not enable_gqa

    ulysess_cp_dim = sdpa_option.ulysess_cp_dim
    if isinstance(ulysess_cp_dim, str):
        ulysess_cp_dim = query.device_mesh.dim_name_index(ulysess_cp_dim)

    if ulysess_cp_dim is not None and query.device_mesh.shape[ulysess_cp_dim] == 1:
        ulysess_cp_dim = None

    ring_cp_dim = sdpa_option.ring_cp_dim
    if isinstance(ring_cp_dim, str):
        ring_cp_dim = query.device_mesh.dim_name_index(ring_cp_dim)

    if ring_cp_dim is not None and query.device_mesh.shape[ring_cp_dim] == 1:
        ring_cp_dim = None

    def remove_sub_split_coordinates(query: Tensor, key: Tensor, value: Tensor):
        old_placements = query.placements
        if key.placements != old_placements or value.placements != old_placements:
            raise RuntimeError("query, key and value have different placements")

        new_placements = []
        for p in old_placements:
            if p.is_shard() and p.has_sub_split_coordinates():
                new_placements.append(Shard(p.get_shard_index()))
            else:
                new_placements.append(p)

        query = query.view(new_placements)
        key = key.view(new_placements)
        value = value.view(new_placements)
        return query, key, value, old_placements

    if not is_causal:
        query, key, value, old_placements = remove_sub_split_coordinates(query, key, value)

    if ulysess_cp_dim is not None:
        assert query.placements[ulysess_cp_dim] == Shard(2)
        assert key.placements[ulysess_cp_dim] == Shard(2)
        assert value.placements[ulysess_cp_dim] == Shard(2)

    if ring_cp_dim is not None:
        assert query.placements[ring_cp_dim] == Shard(2)
        assert key.placements[ring_cp_dim] == Shard(2)
        assert value.placements[ring_cp_dim] == Shard(2)

    if ulysess_cp_dim is not None:
        ulysess_cp_size = query.device_mesh.shape[ulysess_cp_dim]
        Hq = query.shape[1]
        H = key.shape[1]
        assert (
            Hq % ulysess_cp_size == 0 and H % ulysess_cp_size == 0
        ), f"attention heads({Hq}, {H}) cannot be divided evenly by ulysess_cp_size({ulysess_cp_size})"

        ulysess_out_placement = query.placements
        ulysess_out_placement[ulysess_cp_dim] = Shard(1)
        query = query.redistribute(placements=ulysess_out_placement)
        key = key.redistribute(placements=ulysess_out_placement)
        value = value.redistribute(placements=ulysess_out_placement)

    if ring_cp_dim is not None:
        ring_out_placement = query.placements
        ring_out_placement[ring_cp_dim] = Replicate()
        key = key.redistribute(placements=ring_out_placement)
        value = value.redistribute(placements=ring_out_placement)

    out = F._scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        sdpa_option=sdpa_option,
    )

    if ulysess_cp_dim is not None:
        ulysess_out_placement = query.placements
        ulysess_out_placement[ulysess_cp_dim] = Shard(2)
        out = out.redistribute(placements=ulysess_out_placement)

    if not is_causal:
        out = out.view(old_placements)

    return out
