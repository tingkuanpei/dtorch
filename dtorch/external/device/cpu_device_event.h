/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "device_event.h"
#include "dtorch/common/debug.h"

namespace dtorch {
namespace external {
namespace device {

class CpuDeviceEvent : public DeviceEvent {
public:
    CpuDeviceEvent(DeviceEventCreateFlag eventFlag = DeviceEventCreateFlag::kBusyWait)
        : DeviceEvent(DeviceEventType::kCpu), mEventFlag(eventFlag), mMutex(), mCv(), mIsReady(false) {}

    ~CpuDeviceEvent() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(CpuDeviceEvent);

    void Record(DeviceStream& stream) override;

    DTORCH_FORCEINLINE bool Query() override { return mIsReady; }

    void Synchronize() override;

    void StreamWaitEvent(DeviceStream& stream) override;

    void SetNvtxName(const std::string& name) override { IgnoreUnused(name); }

private:
    DeviceEventCreateFlag mEventFlag;
    std::mutex mMutex;
    std::condition_variable mCv;
    std::atomic_bool mIsReady;
};

}  // namespace device
}  // namespace external
}  // namespace dtorch
