"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import os
import asyncio
import atexit
from collections import OrderedDict
from collections.abc import Iterable
import gc
from typing import Union, List, Tuple, Optional
from contextlib import contextmanager

import torch

import dtorch
from dtorch import DeviceMesh, MemoryStats


class GraphOption(dtorch._dtorch_py_api.GraphOption):
    def __init__(self, *, per_device_per_process: Optional[bool] = None) -> None:
        super().__init__()
        self._set_per_device_per_process(per_device_per_process)

    @property
    def per_device_per_process(self) -> Optional[bool]:
        return self._get_per_device_per_process()

    @per_device_per_process.setter
    def per_device_per_process(self, value: Optional[bool]):
        self._set_per_device_per_process(value)

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: "GraphOption") -> bool:
        return self._is_equal(other)


class Graph(dtorch._dtorch_py_api.Graph):
    def __init__(self, *args) -> None:
        if len(args) == 0:
            super().__init__()
        elif len(args) == 1:
            if isinstance(args[0], GraphOption):
                super().__init__(args[0])
            elif isinstance(args[0], dtorch._dtorch_py_api.Graph):
                super().__init__(args[0])
            else:
                raise KeyError(f"Invalid arguments: {args}")
        else:
            raise KeyError(f"Invalid arguments: {args}")

    @property
    def id(self) -> int:
        return self._get_id()

    # each thread have one default graph
    @staticmethod
    def default_graph() -> "Graph":
        return Graph(dtorch._dtorch_py_api._get_thread_local_default_graph())

    def sync(self):
        self._sync()

    def sync_future(self) -> "VoidFutureCollect":
        """Async sync: returns VoidFutureCollect immediately. Call wait() to block."""
        return VoidFutureCollect(self._sync_future())

    def empty_cache(self):
        gc.collect()
        self.sync()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dtorch.nn.functional._empty_cache(self)
        self.sync()

    def get_memory_stats(self, device_mesh: Optional[DeviceMesh] = None, reset_peak: bool = False) -> MemoryStats:
        return MemoryStats(dtorch.nn.functional._get_memory_stats(self, device_mesh, reset_peak))

    @contextmanager
    def device_mesh_guard(self, device_mesh: DeviceMesh):
        original_device_mesh = self._get_default_device_mesh()
        self._set_default_device_mesh(DeviceMesh(device_mesh))

        try:
            yield
        finally:
            self._set_default_device_mesh(original_device_mesh)

    @property
    def default_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(self._get_default_device_mesh())

    @contextmanager
    def dtype_guard(self, dtype: torch.dtype):
        if dtype is None:
            dtype = torch.float32

        original_dtype = self._get_default_dtype()
        self._set_default_dtype(dtype)

        try:
            yield
        finally:
            self._set_default_dtype(original_dtype)

    @property
    def default_dtype(self) -> torch.dtype:
        return self._get_default_dtype()

    def satisfy(self, device_mesh: DeviceMesh) -> bool:
        return self._satisfy(device_mesh)

    def __str__(self) -> str:
        return f"dtorch.Graph with id: {self.id}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: "Graph") -> bool:
        if not isinstance(other, Graph):
            return False
        return self.id == other.id

    def __ne__(self, other):
        return not self == other


class VoidFutureCollect:
    """Python wrapper around C++ VoidFutureCollect for async graph sync.

    Returned by Graph.sync_future(). The user can call wait() to block
    until all devices are synchronized, or check is_ready() to poll without blocking.
    """

    def __init__(self, cpp_future):
        self._cpp_future = cpp_future

    def get(self):
        """Block until all devices are synchronized (consuming)."""
        self._cpp_future.Get()

    def wait(self):
        """Block until all devices are synchronized (non-consuming)."""
        self._cpp_future.Wait()

    def is_ready(self) -> bool:
        """Return True if all devices are synchronized (non-blocking)."""
        return self._cpp_future.IsReady()

    def __await__(self):
        delay = 0.0005
        while not self.is_ready():
            yield from asyncio.sleep(delay).__await__()
        return


# def release_thread_local_default_graph():
#     # Release all tensor, then release thread local default graph
#     gc.collect()
#     dtorch._dtorch_py_api._release_thread_local_default_graph()


# # 1. Default graph may release after pytorch runtime shutting down. So we have to release default graph before
# #    python return
# # 2. Default graph may release after cuda driver shutting down. If default graph contain cuda tensor, it will cause
# #    error. So we have to release default graph before python return
# atexit.register(release_thread_local_default_graph)
