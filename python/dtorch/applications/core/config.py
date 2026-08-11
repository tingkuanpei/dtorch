"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from typing import Union, Optional
from dataclasses import dataclass

from dtorch import DeviceMesh


@dataclass
class CacheConfig:
    return_hidden_states_first: bool = False


@dataclass
class FirstBlockCacheConfig(CacheConfig):
    residual_diff_threshold: float = 0.12
    downsample_factor: int = 1


@dataclass
class TeaCacheConfig(CacheConfig):
    pass


@dataclass
class QuantizeConfig:
    sage_attn_type: Optional[str] = None


@dataclass
class FuseKernelConfig:
    fuse_apply_rotary_emb: bool = False
    fuse_silu_linear_chunk: bool = False
    fuse_layer_norm_mul_add: bool = False


@dataclass
class ExecuteConfig:
    cache_config: Optional[Union[FirstBlockCacheConfig, TeaCacheConfig]] = None
    quant_config: QuantizeConfig = QuantizeConfig()
    fuse_kernel_config: FuseKernelConfig = FuseKernelConfig()

    def check_config(self) -> bool:
        return True
