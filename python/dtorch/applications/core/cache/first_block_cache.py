"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from dataclasses import dataclass
from typing import List, Callable
import inspect

import dtorch

from .cache_base import CacheBase
from dtorch.applications.core.config import FirstBlockCacheConfig


class TransformerBlocksForwardSignature:
    def __init__(self, forward_func: Callable, return_hidden_states_first: bool = True):
        self.return_hidden_states_first = return_hidden_states_first
        self.input_have_hidden_states = False
        self.input_have_encoder_hidden_states = False
        self.input_have_hidden_states_idx = -1
        self.input_have_encoder_hidden_states_idx = -1

        for i, (name, _) in enumerate(inspect.signature(forward_func).parameters.items()):
            if name == "hidden_states":
                self.input_have_hidden_states = True
                self.input_have_hidden_states_idx = i
            elif name == "encoder_hidden_states":
                self.input_have_encoder_hidden_states = True
                self.input_have_encoder_hidden_states_idx = i

        assert self.input_have_hidden_states

    def parse_input(self, *args, **kwargs):
        hidden_states = None
        encoder_hidden_states = None

        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        else:
            assert len(args) > self.input_have_hidden_states_idx
            hidden_states = args[self.input_have_hidden_states_idx]

        if "encoder_hidden_states" in kwargs:
            encoder_hidden_states = kwargs["encoder_hidden_states"]
        else:
            assert len(args) > self.input_have_encoder_hidden_states_idx
            encoder_hidden_states = args[self.input_have_encoder_hidden_states_idx]

        return hidden_states, encoder_hidden_states

    def parse_output(self, block_output):
        hidden_states = None
        encoder_hidden_states = None

        if isinstance(block_output, dtorch.Tensor):
            hidden_states = block_output
        else:
            assert len(block_output) == 2
            hidden_states, encoder_hidden_states = block_output
            if not self.return_hidden_states_first:
                hidden_states, encoder_hidden_states = (
                    encoder_hidden_states,
                    hidden_states,
                )
            if encoder_hidden_states is not None:
                assert isinstance(encoder_hidden_states, dtorch.Tensor)

        assert isinstance(hidden_states, dtorch.Tensor)
        return hidden_states, encoder_hidden_states

    def return_hidden_states(self, hidden_states, encoder_hidden_states):
        if encoder_hidden_states is None:
            return hidden_states
        elif self.return_hidden_states_first:
            return hidden_states, encoder_hidden_states
        else:
            return encoder_hidden_states, hidden_states

    def return_hidden_states_directly(self, *args, **kwargs):
        hidden_states, encoder_hidden_states = self.parse_input(*args, **kwargs)
        return self.return_hidden_states(hidden_states, encoder_hidden_states)


class FirstBlockCache(CacheBase):
    def __init__(self, pipe_name: str, config: FirstBlockCacheConfig):
        super(FirstBlockCache, self).__init__(pipe_name, config)
        self.can_use_cache = False
        # hidden_states residual between first block input and output
        self.prev_first_hidden_states_residual = None
        # hidden_states and encoder_hidden_states residual between first block output and last block output
        self.prev_total_hidden_states_residual = None
        self.prev_total_encoder_hidden_states_residual = None
        # hidden_states and encoder_hidden_states of first block output
        self.first_block_output_hidden_states = None
        self.first_block_output_encoder_hidden_states = None

    def register_transformer_blocks_forward_hook(self, transformer_blocks: List[dtorch.nn.Module]):
        assert len(transformer_blocks) > 0

        transformer_blocks[0].forward = self.first_block_forward_hook(transformer_blocks[0].forward)

        for block in transformer_blocks[1:-1]:
            block.forward = self.mid_block_forward(block.forward)

        transformer_blocks[-1].forward = self.last_block_forward(transformer_blocks[-1].forward)

    def first_block_forward_hook(self, origin_forward):
        signature = TransformerBlocksForwardSignature(origin_forward, self.config.return_hidden_states_first)

        def forward_hook(*args, **kwargs):
            first_block_output = origin_forward(*args, **kwargs)

            input_hidden_states, _ = signature.parse_input(*args, **kwargs)
            output_hidden_states, output_encoder_hidden_states = signature.parse_output(first_block_output)

            first_hidden_states_residual = output_hidden_states - input_hidden_states
            self.can_use_cache = self.compute_can_use_cache(first_hidden_states_residual)

            if self.can_use_cache:
                (hidden_states, encoder_hidden_states,) = self.apply_prev_hidden_states_residual(
                    output_hidden_states,
                    output_encoder_hidden_states,
                )
                return signature.return_hidden_states(hidden_states, encoder_hidden_states)
            else:
                self.set_first_hidden_states_residual(first_hidden_states_residual)
                self.first_block_output_hidden_states = output_hidden_states
                self.first_block_output_encoder_hidden_states = output_encoder_hidden_states
                return first_block_output

        return forward_hook

    def mid_block_forward(self, origin_forward):
        signature = TransformerBlocksForwardSignature(origin_forward, self.config.return_hidden_states_first)

        def forward_hook(*args, **kwargs):
            if self.can_use_cache:
                return signature.return_hidden_states_directly(*args, **kwargs)
            else:
                return origin_forward(*args, **kwargs)

        return forward_hook

    def last_block_forward(self, origin_forward):
        signature = TransformerBlocksForwardSignature(origin_forward, self.config.return_hidden_states_first)

        def forward_hook(*args, **kwargs):
            if self.can_use_cache:
                return signature.return_hidden_states_directly(*args, **kwargs)
            else:
                last_block_output = origin_forward(*args, **kwargs)
                (
                    last_block_output_hidden_states,
                    last_block_output_encoder_hidden_states,
                ) = signature.parse_output(last_block_output)

                self.prev_total_hidden_states_residual = (
                    last_block_output_hidden_states - self.first_block_output_hidden_states
                )
                self.prev_total_encoder_hidden_states_residual = (
                    (last_block_output_encoder_hidden_states - self.first_block_output_encoder_hidden_states)
                    if last_block_output_encoder_hidden_states is not None
                    else None
                )
                self.first_block_output_hidden_states = None
                self.first_block_output_encoder_hidden_states = None

                return last_block_output

        return forward_hook

    def set_first_hidden_states_residual(self, first_hidden_states_residual):
        downsample_factor = self.config.downsample_factor
        if downsample_factor > 1:
            first_hidden_states_residual = first_hidden_states_residual[..., ::downsample_factor]
            first_hidden_states_residual = first_hidden_states_residual.contiguous()
        self.prev_first_hidden_states_residual = first_hidden_states_residual

    def compute_can_use_cache(self, first_hidden_states_residual: dtorch.Tensor):
        threshold = self.config.residual_diff_threshold
        if threshold <= 0.0:
            return False

        downsample_factor = self.config.downsample_factor
        if downsample_factor > 1:
            first_hidden_states_residual = first_hidden_states_residual[..., ::downsample_factor]

        can_use_cache = self.prev_first_hidden_states_residual is not None and self.are_two_tensors_similar(
            self.prev_first_hidden_states_residual,
            first_hidden_states_residual,
            threshold=threshold,
        )
        return can_use_cache

    @staticmethod
    def are_two_tensors_similar(t1: dtorch.Tensor, t2: dtorch.Tensor, *, threshold: float):
        if threshold <= 0.0:
            return False

        if t1.shape != t2.shape:
            return False

        mean_diff = (t1 - t2).abs().mean()
        mean_t1 = t1.abs().mean()
        diff = mean_diff / mean_t1
        return diff.item() < threshold

    def apply_prev_hidden_states_residual(self, hidden_states, encoder_hidden_states=None):
        hidden_states_residual = self.prev_total_hidden_states_residual
        assert hidden_states_residual is not None, "hidden_states_residual must be set before"
        hidden_states = hidden_states_residual + hidden_states
        hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            if self.prev_total_encoder_hidden_states_residual is None:
                encoder_hidden_states = None
            else:
                encoder_hidden_states_residual = self.prev_total_encoder_hidden_states_residual
                encoder_hidden_states = encoder_hidden_states_residual + encoder_hidden_states
                encoder_hidden_states = encoder_hidden_states.contiguous()

        return hidden_states, encoder_hidden_states
