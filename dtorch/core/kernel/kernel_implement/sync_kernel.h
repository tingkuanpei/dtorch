/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/kernel/kernel.h"

namespace dtorch {
namespace core {

// ============================================================
// class SyncKernel
// ============================================================
//
// Per-device kernel for SyncOp. Runs on each target device's kernel
// stream. When Compute() is called, all previous kernels on this
// stream have completed. SyncKernel synchronizes the device (GPU:
// event + BoostAsioThreadPool; CPU: direct) and then fulfills the
// VoidPromise.

class SyncKernel : public Kernel {
public:
    SyncKernel(const KernelCreateCtx& ctx) : Kernel(ctx) {}

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) override;
};

}  // namespace core
}  // namespace dtorch
