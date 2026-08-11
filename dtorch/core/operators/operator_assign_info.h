/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

struct OperatorAssignInfo {
public:
    OperatorAssignInfo() : mMaxGpuId(-1), mStreamKeySet() {}

    DTORCH_FORCEINLINE void Insert(KernelStreamKey& streamKey) {
        mStreamKeySet.insert(streamKey);
        if (streamKey.device.deviceKind == DeviceKind::kGpu && streamKey.device.deviceId > mMaxGpuId) {
            mMaxGpuId = streamKey.device.deviceId;
        }
    }

    DTORCH_FORCEINLINE size_t NumKernelForThisOp() const noexcept { return mStreamKeySet.size(); }

    DTORCH_FORCEINLINE int64_t MaxGpuId() const noexcept { return mMaxGpuId; }

    DTORCH_FORCEINLINE const StreamKeySet& GetStreamKeySet() const noexcept { return mStreamKeySet; }

    std::string ToString() const {
        std::stringstream ss;
        ss << "OperatorAssignInfo[";
        for (const auto& it : mStreamKeySet) {
            ss << it << ", ";
        }
        ss << "]";
        return ss.str();
    }

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const OperatorAssignInfo& info) {
        os << info.ToString();
        return os;
    }

private:
    int64_t mMaxGpuId;
    StreamKeySet mStreamKeySet;
};

}  // namespace core
}  // namespace dtorch
