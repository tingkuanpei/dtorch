/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "sync_op.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void SyncOp::InferOperatorAssignInfo() {
    DDebugAssert(mOperatorAssignInfo.NumKernelForThisOp() == 0);

    const auto& param = GetOpParam<SyncParam>();
    for (const auto& device : param.syncDevices) {
        KernelStreamKey streamKey;
        streamKey.Init(device, KernelStreamType::kCompute);
        mOperatorAssignInfo.Insert(streamKey);
    }
}

std::string SyncOp::GetDescribeString() const {
    const auto& param = GetOpParam<SyncParam>();
    std::stringstream ss;
    ss << "SyncOp[devices=";
    for (size_t i = 0; i < param.syncDevices.size(); i++) {
        if (i > 0) ss << ",";
        ss << param.syncDevices[i];
    }
    ss << "]";
    return ss.str();
}

}  // namespace core
}  // namespace dtorch
