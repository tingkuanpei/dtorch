/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/external/cuda/nvtx_profiler.h"
#include "kernel_stream.h"

namespace dtorch {
namespace core {

class CpuKernelStream : public KernelStream {
public:
    static DeviceKind SupportDeviceKind() { return DeviceKind::kCpu; }

public:
    CpuKernelStream(int64_t deviceId, KernelStreamType streamType, bool isAsync)
        : KernelStream(Device(DeviceKind::kCpu, deviceId), streamType, isAsync) {
        Init();
    }

    void InitInAsyncThread() override {
        std::string threadName = "DTorchCpuStreamForDevice" + std::to_string(GetDevice().deviceId);
        UNvtxNameOsThread(threadName);
    }
};

}  // namespace core
}  // namespace dtorch
