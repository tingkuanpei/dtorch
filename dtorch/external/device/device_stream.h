/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

#include "dtorch/common/debug.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace external {
namespace device {

class DeviceStream {
public:
    static DeviceStream GetCurrentStream(const core::Device& localDevice);

public:
    DeviceStream(const core::Device& localDevice = core::Device::GetDefaultCpuDevice())
        : mLocalDevice(localDevice), mTorchCudaStream(nullptr) {
        DDebugAssert(localDevice.deviceKind == core::DeviceKind::kCpu);
    }

    DeviceStream(const core::Device& device, std::shared_ptr<at::cuda::CUDAStream> cudaStream)
        : mLocalDevice(device), mTorchCudaStream(cudaStream) {
        DDebugAssert(device.deviceKind == core::DeviceKind::kGpu && cudaStream != nullptr);
    }

    DTORCH_FORCEINLINE const core::Device& GetLocalDevice() const noexcept { return mLocalDevice; }

    DTORCH_FORCEINLINE core::DeviceKind GetDeviceKind() const noexcept { return mLocalDevice.deviceKind; }

    DTORCH_FORCEINLINE std::shared_ptr<at::cuda::CUDAStream> GetTorchCudaStream() noexcept {
        DDebugAssert(mLocalDevice.deviceKind == core::DeviceKind::kGpu && mTorchCudaStream != nullptr);
        return mTorchCudaStream;
    }

protected:
    core::Device mLocalDevice;
    std::shared_ptr<at::cuda::CUDAStream> mTorchCudaStream;
};

}  // namespace device
}  // namespace external
}  // namespace dtorch
