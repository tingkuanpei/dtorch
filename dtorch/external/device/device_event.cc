/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "device_event.h"

#include <memory>
#include <stdexcept>

#include "dtorch/api/cpp/device.h"
#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/external/device/cpu_device_event.h"

#if DTORCH_WITH_CUDA
#include "dtorch/external/device/cuda_device_event.h"
#endif

namespace dtorch {
namespace external {
namespace device {

std::unique_ptr<DeviceEvent> DeviceEvent::CreateDeviceEvent(core::DeviceKind deviceKind,
                                                            DeviceEventCreateFlag eventCreateFlag) {
    if (deviceKind == core::DeviceKind::kCpu) {
        return std::make_unique<CpuDeviceEvent>(eventCreateFlag);
    }
#if DTORCH_WITH_CUDA
    else if (deviceKind == core::DeviceKind::kGpu) {
        DDebugAssert(deviceKind == core::DeviceKind::kGpu);
        return std::make_unique<CudaDeviceEvent>(eventCreateFlag);
    }
#endif
    else {
        throw std::invalid_argument("Unsupport device: " + core::DeviceKindToString(deviceKind));
    }
}

}  // namespace device
}  // namespace external
}  // namespace dtorch
