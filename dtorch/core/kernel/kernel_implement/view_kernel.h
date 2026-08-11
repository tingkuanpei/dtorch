/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/kernel/kernel.h"

namespace dtorch {
namespace core {

class ViewKernel : public Kernel {
public:
    ViewKernel(const KernelCreateCtx& ctx) : Kernel(ctx) {}

    DTORCH_DEFAULT_COPY_AND_MOVE(ViewKernel);

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) override;
};

}  // namespace core
}  // namespace dtorch
