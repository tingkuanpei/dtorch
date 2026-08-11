"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from contextlib import contextmanager
from typing import Union
import functools

import torch

import dtorch
from dtorch import Graph
import dtorch.nn.functional as F


def range_push(msg: str, graph: Graph = Graph.default_graph()):
    F._nvtx_range_push(graph, msg)


def range_pop(graph: Graph = Graph.default_graph()):
    F._nvtx_range_pop(graph)


def mark(msg: str, graph: Graph = Graph.default_graph()):
    F._nvtx_mark(graph, msg)


@contextmanager
def range(msg: str, graph: Graph = Graph.default_graph(), *args, **kwargs):
    range_push(msg.format(*args, **kwargs), graph=graph)
    try:
        yield
    finally:
        range_pop(graph=graph)


def module_register_nvtx(module: Union[torch.nn.Module, dtorch.nn.Module], tag: str = "module"):
    def module_forward_decorator(func, module, tag):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tag_split = tag.split(".")
            nvtx_msg = tag_split[-1]
            if nvtx_msg.isdigit() and len(tag_split) >= 2:
                nvtx_msg = ".".join(tag_split[-2:])

            assert isinstance(module, torch.nn.Module) or isinstance(module, dtorch.nn.Module)
            if isinstance(module, dtorch.nn.Module):
                with dtorch.cuda.nvtx.range(nvtx_msg):
                    return func(*args, **kwargs)
            else:
                with torch.cuda.nvtx.range(nvtx_msg):
                    return func(*args, **kwargs)

        return wrapper

    for name, m in module.named_modules(remove_duplicate=True):
        name = tag if name == "" else name
        assert name != "" and name is not None
        m.forward = module_forward_decorator(m.forward, m, name)
