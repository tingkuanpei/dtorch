/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>

#include "device_event.h"
#include "dtorch/external/cuda/cuda_event.h"

namespace dtorch {
namespace external {
namespace device {

class CudaDeviceEvent : public DeviceEvent {
public:
    CudaDeviceEvent(DeviceEventCreateFlag eventFlag = DeviceEventCreateFlag::kBusyWait);

    CudaDeviceEvent(const std::string& ipcEventHandle);

    ~CudaDeviceEvent() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(CudaDeviceEvent);

    void Record(DeviceStream& stream) override;

    DTORCH_FORCEINLINE bool Query() override { return mCudaEventRecorded && mCudaEvent.Query(); }

    void Synchronize() override;

    DTORCH_FORCEINLINE external::cuda::CudaEvent GetNative() { return mCudaEvent; }

    void StreamWaitEvent(DeviceStream& stream) override;

    void SetNvtxName(const std::string& name) override;

private:
    void WaitUntilEventRecoed();

private:
    std::mutex mMutex;
    std::condition_variable mCv;
    std::atomic_bool mCudaEventRecorded;
    DeviceEventCreateFlag mEventFlag;
    external::cuda::CudaEvent mCudaEvent;
};

}  // namespace device
}  // namespace external
}  // namespace dtorch
