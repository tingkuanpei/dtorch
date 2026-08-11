/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "cuda_device_event.h"

#include <c10/cuda/CUDAStream.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/type_cast.h"
#include "dtorch/external/cuda/cuda_device.h"
#include "dtorch/external/cuda/nvtx_profiler.h"

namespace dtorch {
namespace external {
namespace device {

CudaDeviceEvent::CudaDeviceEvent(DeviceEventCreateFlag eventFlag)
    : DeviceEvent(DeviceEventType::kCuda),
      mMutex(),
      mCv(),
      mCudaEventRecorded(false),
      mEventFlag(eventFlag),
      mCudaEvent() {}

CudaDeviceEvent::CudaDeviceEvent(const std::string& ipcEventHandle)
    : DeviceEvent(DeviceEventType::kCuda),
      mMutex(),
      mCv(),
      mCudaEventRecorded(false),
      mEventFlag(DeviceEventCreateFlag::kBusyWait),
      mCudaEvent() {
    // Not need external::cuda::CudaDeviceGuard guard(deviceId);
    mCudaEvent.OpenEventHandle(ipcEventHandle);
    mCudaEventRecorded = true;
}

void CudaDeviceEvent::Record(DeviceStream& stream) {
    DDebugAssert(stream.GetDeviceKind() == core::DeviceKind::kGpu);
    DAlwaysAssert(mCudaEventRecorded == false);

    std::unique_lock<std::mutex> lock(mMutex);
    const int deviceId = stream.GetLocalDevice().deviceId;
    external::cuda::CudaDeviceGuard guard(deviceId);

    // event and stream must be on the same CUDA context
    switch (mEventFlag) {
        case DeviceEventCreateFlag::kBusyWait:
            mCudaEvent.Create(cudaEventDefault | cudaEventDisableTiming);
            break;
        case DeviceEventCreateFlag::kBlockingSync:
            mCudaEvent.Create(cudaEventBlockingSync | cudaEventDisableTiming);
            break;
        case DeviceEventCreateFlag::kInterprocess:
            mCudaEvent.Create(cudaEventInterprocess | cudaEventDisableTiming);
            break;
        default:
            DUnimplemented();
            break;
    }

    mCudaEvent.Record(stream.GetTorchCudaStream()->stream());
    mCudaEventRecorded = true;
    lock.unlock();
    mCv.notify_all();
}

void CudaDeviceEvent::WaitUntilEventRecoed() {
    if (mCudaEventRecorded == false) {
        std::unique_lock<std::mutex> lock(mMutex);
        if (mCudaEventRecorded == false) {
            mCv.wait(lock, [&] { return mCudaEventRecorded == true; });
        }
    }
}

void CudaDeviceEvent::Synchronize() {
    WaitUntilEventRecoed();
    mCudaEvent.Synchronize();
}

void CudaDeviceEvent::StreamWaitEvent(DeviceStream& stream) {
    if (stream.GetDeviceKind() == core::DeviceKind::kCpu) {
        Synchronize();
    } else {
        DDebugAssert(stream.GetDeviceKind() == core::DeviceKind::kGpu);

        WaitUntilEventRecoed();
        external::cuda::CudaDeviceGuard guard(stream.GetLocalDevice().deviceId);
        GetNative().StreamWaitEvent(stream.GetTorchCudaStream()->stream());
    }
}

void CudaDeviceEvent::SetNvtxName(const std::string& name) {
    external::cuda::NvtxProfile::NameEvent(mCudaEvent.Get(), name);
}

}  // namespace device
}  // namespace external
}  // namespace dtorch
