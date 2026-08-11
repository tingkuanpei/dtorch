/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/kernel/kernel.h"

namespace dtorch {
namespace core {

class ReduceKernel : public Kernel {
public:
    ReduceKernel(const KernelCreateCtx& ctx) : Kernel(ctx), mThreadGroupManager(ctx.threadGroupManager) {
        DDebugAssert(mThreadGroupManager != nullptr);
    }

    DTORCH_DEFAULT_COPY_AND_MOVE(ReduceKernel);

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) override;

    void ComputeLocalResult(const TorchTensorOptArray& inputs, TorchTensorArray& outputs);

private:
    communication::ThreadGroupManager* mThreadGroupManager;
};

}  // namespace core
}  // namespace dtorch
