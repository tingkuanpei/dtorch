/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <mutex>
#include <unordered_map>

#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/kernel/kernel.h"

namespace dtorch {
namespace core {

class ConvertKernel : public Kernel {
public:
    ConvertKernel(const KernelCreateCtx& ctx) : Kernel(ctx), mThreadGroupManager(ctx.threadGroupManager) {
        DDebugAssert(mThreadGroupManager != nullptr);
    }

    DTORCH_DEFAULT_COPY_AND_MOVE(ConvertKernel);

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) override;

private:
    DTORCH_FORCEINLINE DataKind GetSrcDataKind() const { return mOp->OperandX()->GetDataKind(); }

    DTORCH_FORCEINLINE const DeviceMesh& GetSrcDeviceMesh() const { return mOp->OperandX()->GetDeviceMesh(); }

    DTORCH_FORCEINLINE const PlacementSeq& GetSrcPlacementSeq() const { return mOp->OperandX()->GetPlacementSeq(); }

    DTORCH_FORCEINLINE DataKind GetDestDataKind() const { return mOp->OperandY()->GetDataKind(); }

    DTORCH_FORCEINLINE const DeviceMesh& GetDestDeviceMesh() const { return mOp->OperandY()->GetDeviceMesh(); }

    DTORCH_FORCEINLINE const PlacementSeq& GetDestPlacementSeq() const { return mOp->OperandY()->GetPlacementSeq(); }

    void ConvertDeviceMesh(const TorchTensorOptArray& inputs, TorchTensorArray& outputs);

    void ConvertPlacements(const TorchTensorOptArray& inputs, TorchTensorArray& outputs);

    // Convert dtype for dtensor and convert device and dtype for local tensor
    void ConvertDeviceAndDataKind(const TorchTensorOptArray& inputs, TorchTensorArray& outputs);

    void ScatherTensor(const TorchTensorOptArray& inputs, TorchTensorArray& outputs);

    void GatherTensor(const TorchTensorOptArray& inputs, TorchTensorArray& outputs);

private:
    communication::ThreadGroupManager* mThreadGroupManager;
};

}  // namespace core
}  // namespace dtorch
