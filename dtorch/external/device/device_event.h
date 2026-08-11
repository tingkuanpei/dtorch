/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/type.h"
#include "dtorch/external/device/device_stream.h"

namespace dtorch {
namespace external {
namespace device {

enum class DeviceEventCreateFlag {
    kBusyWait = 0,
    kBlockingSync,
    kInterprocess,
};

enum class DeviceEventType {
    kCpu = 0,
    kCuda,
    kCount,
};

class DeviceEvent {
public:
    static std::unique_ptr<DeviceEvent> CreateDeviceEvent(
        core::DeviceKind deviceKind, DeviceEventCreateFlag eventCreateFlag = DeviceEventCreateFlag::kBusyWait);

public:
    DeviceEvent(DeviceEventType type) : mEventType(type) {}

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(DeviceEvent);

    DeviceEventType GetEventType() const noexcept { return mEventType; }

    virtual ~DeviceEvent() = default;

    virtual void Record(DeviceStream& stream) = 0;

    virtual bool Query() = 0;

    virtual void Synchronize() = 0;

    virtual void StreamWaitEvent(DeviceStream& stream) = 0;

    virtual void SetNvtxName(const std::string& name) = 0;

private:
    DeviceEventType mEventType;
};

}  // namespace device
}  // namespace external
}  // namespace dtorch
