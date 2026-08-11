"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import dtorch
from dtorch import Tensor
from .module import Module
from .. import functional as F


class Dropout(Module):
    p: float

    def __init__(
        self,
        p: float = 0.5,
    ) -> None:
        super(Dropout, self).__init__()
        # TODO: support p == 1.0f
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be [0, 1), " "but got {}".format(p))
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, p=self.p, training=self.training)

    def extra_repr(self) -> str:
        return "p={}".format(
            self.p,
        )
