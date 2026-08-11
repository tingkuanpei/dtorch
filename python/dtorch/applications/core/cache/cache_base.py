"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import dtorch
from dtorch.applications.core.config import (
    CacheConfig,
    TeaCacheConfig,
    FirstBlockCacheConfig,
)


class CacheBase(ABC):
    def __init__(self, pipe_name: str, config: CacheConfig):
        self.pipe_name = pipe_name
        self.config = config

    @abstractmethod
    def register_transformer_blocks_forward_hook(self, transformer_blocks: List[dtorch.nn.Module]):
        pass


def enable_cache(
    pipe_name: str,
    cache_config: CacheConfig,
    transformer_blocks: List[dtorch.nn.Module],
) -> "CacheBase":
    if cache_config is None or len(transformer_blocks) == 0:
        return None

    from .first_block_cache import FirstBlockCache

    # from .tea_cache import TeaCache

    cache_class = None
    if isinstance(cache_config, FirstBlockCacheConfig):
        cache_class = FirstBlockCache
    else:
        raise KeyError(f"Unsupport cache config: {cache_config}")

    cache_context = cache_class(pipe_name, cache_config)
    cache_context.register_transformer_blocks_forward_hook(transformer_blocks)
    return cache_context
