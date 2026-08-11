"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

from dtorch import Tensor


class Parameter(Tensor):
    def __init__(self, tensor: Tensor = None) -> None:
        super().__init__(tensor)


class Buffer(Tensor):
    def __init__(self, tensor: Tensor = None, *, persistent=True) -> None:
        super().__init__(tensor)
        self.persistent = persistent
