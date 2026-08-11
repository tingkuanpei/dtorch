/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "copy_kernel.h"

#include <memory>

#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/communication/tensor_store/file_tensor_store.h"
#include "dtorch/core/kernel_stream/cuda_kernel_stream.h"

namespace dtorch {
namespace core {

void CopyKernel::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& /*outputs*/) {
    DDebugAssert(inputs.size() == 2);
    if (mLocalDevice.deviceKind == DeviceKind::kGpu) {
        TorchCudaStreamGuarantee::CheckStream(*(GetDeviceStream().GetTorchCudaStream()));
    }

    if (mNumKernelForThisOp == mOp->OperandA()->GetDeviceMesh().Count()) {
        DAlwaysAssert(mOp->OperandA()->GetDeviceMesh() == mOp->OperandB()->GetDeviceMesh());
        inputs[0].value().copy_(inputs[1].value());
    } else {
        DDebugAssert(!mOp->OperandA()->IsDistributed() && !mOp->OperandB()->IsDistributed());
        auto tensorStore = GetTensorStore();
        DeviceStream deviceStream = GetDeviceStream();
        const std::string key = "tensor";

        if (!GlobalDeviceInOperand(mOp->OperandA())) {
            DDebugAssert(GlobalDeviceInOperand(mOp->OperandB()));
            DDebugAssert(inputs[1].has_value());
            DeviceKind destGetDeviceKind = mOp->OperandA()->GetDeviceKind();
            tensorStore->SrcSet(key, inputs[1].value(), deviceStream, 1, destGetDeviceKind);
            tensorStore->SrcWaitUntilGetFinished(key, deviceStream);
        } else {
            DDebugAssert(inputs[0].has_value());
            inputs[0].value().copy_(tensorStore->DestGet(key, deviceStream));
            tensorStore->DestFinishGet(key, deviceStream);
        }
    }
}

}  // namespace core
}  // namespace dtorch
