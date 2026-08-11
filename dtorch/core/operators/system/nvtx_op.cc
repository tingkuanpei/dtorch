/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "nvtx_op.h"

#include "dtorch/common/config.h"
#if DTORCH_WITH_CUDA
#include "dtorch/external/cuda/nvtx_profiler.h"
#endif

namespace dtorch {
namespace core {

void NvtxOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 0 && GetOutputSize() == 0);
    const auto& param = GetOpParam<NvtxParam>();

    if ((param.nvtxType == NvtxType::kRangePush || param.nvtxType == NvtxType::kMark) && param.message.empty()) {
        throw std::invalid_argument("Message is empty");
    }
}

void NvtxOp::Compute(const TorchTensorOptArray& /*inputs*/, TorchTensorArray& /*outputs*/) const {
#if DTORCH_WITH_CUDA
    const auto& param = GetOpParam<NvtxParam>();
    switch (param.nvtxType) {
        case NvtxType::kRangePush:
            external::cuda::NvtxProfile::RangePush(param.message);
            break;
        case NvtxType::kRangePop:
            external::cuda::NvtxProfile::RangePop();
            break;
        case NvtxType::kMark:
            external::cuda::NvtxProfile::Mark(param.message);
            break;
        default:
            DUnimplemented();
            break;
    }
#endif
}

void NvtxOp::InferOperatorAssignInfo() {
    DDebugAssert(mOperatorAssignInfo.NumKernelForThisOp() == 0);

    for (const auto& device : GetOpParam<NvtxParam>().deviceMesh.ToDeviceVec()) {
        KernelStreamKey streamKey;
        streamKey.Init(device, KernelStreamType::kCompute);
        mOperatorAssignInfo.Insert(streamKey);
    }
}

}  // namespace core
}  // namespace dtorch
