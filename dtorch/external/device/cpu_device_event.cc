/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "cpu_device_event.h"

#include <chrono>
#include <thread>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace external {
namespace device {

void CpuDeviceEvent::Record(DeviceStream& stream) {
    DDebugAssert(stream.GetDeviceKind() == core::DeviceKind::kCpu);
    DAlwaysAssert(mIsReady == false);

    std::unique_lock<std::mutex> lock(mMutex);
    mIsReady = true;

    if (mEventFlag == DeviceEventCreateFlag::kBlockingSync) {
        lock.unlock();
        mCv.notify_all();
    }
}

void CpuDeviceEvent::Synchronize() {
    if (mEventFlag == DeviceEventCreateFlag::kBusyWait) {
        while (!mIsReady) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(100us);
        }
    } else if (mEventFlag == DeviceEventCreateFlag::kBlockingSync) {
        if (!mIsReady) {
            std::unique_lock<std::mutex> lock(mMutex);
            if (!mIsReady) {
                mCv.wait(lock, [&] { return mIsReady == true; });
            }
        }
    } else {
        DUnimplemented();
    }
}

void CpuDeviceEvent::StreamWaitEvent(DeviceStream& /*stream*/) { Synchronize(); }

}  // namespace device
}  // namespace external
}  // namespace dtorch
