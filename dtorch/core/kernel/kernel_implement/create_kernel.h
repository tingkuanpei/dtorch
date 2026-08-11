/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/kernel/kernel.h"

namespace dtorch {
namespace core {

class CreateKernel : public Kernel {
public:
    CreateKernel(const KernelCreateCtx& ctx) : Kernel(ctx) {}

    DTORCH_DEFAULT_COPY_AND_MOVE(CreateKernel);

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) override;
};

}  // namespace core
}  // namespace dtorch
