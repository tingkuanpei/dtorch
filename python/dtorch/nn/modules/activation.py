"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import dtorch
from dtorch import Tensor
from .module import Module
from .. import functional as F


class ReLU(Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class LeakyReLU(Module):
    __constants__ = ["negative_slope", "inplace"]
    negative_slope: float
    inplace: bool

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"negative_slope={self.negative_slope}{inplace_str}"


class ELU(Module):
    __constants__ = ["alpha", "inplace"]
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"alpha={self.alpha}{inplace_str}"


class GELU(Module):
    __constants__ = ["approximate"]
    approximate: str

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={repr(self.approximate)}"


class SiLU(Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Softmax(Module):
    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim: int = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
