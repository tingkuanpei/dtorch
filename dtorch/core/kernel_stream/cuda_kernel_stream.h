/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

#include "dtorch/common/debug.h"
#include "dtorch/external/cuda/cuda_device.h"
#include "dtorch/external/torch/torch_util.h"
#include "kernel_stream.h"

namespace dtorch {
namespace core {

class CudaKernelStream : public KernelStream {
public:
    static DeviceKind SupportDeviceKind() { return DeviceKind::kGpu; }

public:
    CudaKernelStream(int64_t deviceId, KernelStreamType streamType, bool isAsync)
        : KernelStream(Device(DeviceKind::kGpu, deviceId), streamType, isAsync), mStream() {
        DAlwaysAssert(deviceId >= 0 && deviceId < external::cuda::CudaDevice::GetDeviceCount());
        Init();
    }

    void InitInAsyncThread() override;

    void SyncImp() override;

    DTORCH_FORCEINLINE at::cuda::CUDAStream& GetTorchCudaStream() noexcept { return *(mStream.get()); }

private:
    std::shared_ptr<at::cuda::CUDAStream> mStream;
};

class TorchCudaStreamGuarantee {
public:
    static void SetStream(at::cuda::CUDAStream& stream);

    static void CheckStream(const at::cuda::CUDAStream& stream);

private:
    static thread_local std::shared_ptr<at::cuda::CUDAStream> mStream;
};

}  // namespace core
}  // namespace dtorch
