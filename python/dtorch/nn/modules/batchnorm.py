"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import torch

import dtorch
from dtorch import Tensor, OperatorFormat
from dtorch.nn.parameter import Parameter
from .module import Module
from .. import functional as F


# training and track_running_stats status
# https://www.zhihu.com/question/282672547
class BatchNorm2d(Module):
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    operator_format: OperatorFormat

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        is_nchw: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # operator format
        self.operator_format = OperatorFormat.nchw
        if not is_nchw:
            self.operator_format = OperatorFormat.nhwc

        factory_kwargs = {"device": device, "dtype": dtype}
        if self.affine:
            self.weight = Parameter(dtorch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(dtorch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", dtorch.zeros(num_features, **factory_kwargs))
            self.register_buffer("running_var", dtorch.ones(num_features, **factory_kwargs))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            self.momentum,
            self.eps,
            self.operator_format,
        )
