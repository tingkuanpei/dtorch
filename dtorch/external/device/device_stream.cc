/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "device_stream.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace external {
namespace device {

DeviceStream DeviceStream::GetCurrentStream(const core::Device& localDevice) {
    if (localDevice.deviceKind == core::DeviceKind::kCpu) {
        return DeviceStream(localDevice);
    } else if (localDevice.deviceKind == core::DeviceKind::kGpu) {
        c10::cuda::CUDAGuard guard(localDevice.deviceId);
        std::shared_ptr<at::cuda::CUDAStream> cudaStream =
            std::make_shared<at::cuda::CUDAStream>(at::cuda::getCurrentCUDAStream(localDevice.deviceId));
        DAlwaysAssert(cudaStream->device_index() == localDevice.deviceId);
        return DeviceStream(localDevice, cudaStream);
    } else {
        DLogError() << "Unsupport device kind: " << localDevice.deviceKind;
        DUnimplemented();
    }
    return DeviceStream(localDevice);
}

}  // namespace device
}  // namespace external
}  // namespace dtorch
