/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "device.h"

#include <array>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/string.h"
#include "dtorch/external/torch/torch_util.h"
#if DTORCH_WITH_CUDA
#include "dtorch/external/cuda/cuda_device.h"
#endif

namespace dtorch {
namespace api {
namespace cpp {

DeviceKind DeviceKindFromString(const std::string& deviceStr) {
    std::string lowerCaseDeviceStr = deviceStr;
    String::ToLower(lowerCaseDeviceStr);
    if (lowerCaseDeviceStr == "cuda") {
        lowerCaseDeviceStr = "gpu";
    }

    static const std::unordered_map<std::string, DeviceKind> kDeviceKindMap{
        {"cpu", DeviceKind::kCpu},
        {"gpu", DeviceKind::kGpu},
    };
    DDebugAssert(static_cast<int64_t>(kDeviceKindMap.size()) == EnumAsInteger(DeviceKind::kCount));

    DeviceKind result = DeviceKind::kCpu;
    if (kDeviceKindMap.count(lowerCaseDeviceStr) > 0) {
        result = kDeviceKindMap.at(lowerCaseDeviceStr);
    } else {
        throw std::invalid_argument("Unsupported device str: " + deviceStr);
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, DeviceKind deviceKind) {
    os << DeviceKindToString(deviceKind);
    return os;
}

std::string DeviceKindToString(DeviceKind deviceKind) {
    static const std::array<std::string, 2> kStringMap = {"cpu", "gpu"};
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(DeviceKind::kCount),
                  "Device kind size not equal");

    if (deviceKind == DeviceKind::kCount) {
        DLogError() << "device Kind invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(deviceKind)];
}

Device::Device(torch::Device torchDevice) : deviceKind(DeviceKind::kCpu), deviceId(0) {
    *this = external::torch::TorchUtil::ToDevice(torchDevice);
}

Device::Device(const std::string& str) : deviceKind(DeviceKind::kCpu), deviceId(0) {
    std::string deviceStr = "cpu";
    int deviceId = 0;
    std::stringstream ss;
    ss << "Invalid device string: " << str;

    try {
        std::vector<std::string> splitStr = String::Split(str, ":");
        size_t splitSize = splitStr.size();
        if (splitSize > 3) {
            throw std::invalid_argument(ss.str());
        }

        if (splitSize == 1) {
            deviceStr = splitStr[0];
        } else if (splitSize == 2) {
            deviceStr = splitStr[0];
            deviceId = std::stoi(splitStr[1]);
        }
    } catch (std::exception& e) {
        throw std::invalid_argument(ss.str());
    }

    this->deviceKind = DeviceKindFromString(deviceStr);
    this->deviceId = deviceId;
}

bool Device::IsAvailable(DeviceKind deviceKind) noexcept {
    if (deviceKind == DeviceKind::kCpu) {
        return true;
    } else if (deviceKind == DeviceKind::kGpu) {
#if DTORCH_WITH_CUDA
        return Device::DeviceCount(DeviceKind::kGpu) > 0;
#endif
    }
    return false;
}

size_t Device::DeviceCount(DeviceKind deviceKind) noexcept {
    if (deviceKind == DeviceKind::kCpu) {
        return 1;
    } else if (deviceKind == DeviceKind::kGpu) {
#if DTORCH_WITH_CUDA
        return external::cuda::CudaDevice::GetDeviceCount();
#else
        return 0;
#endif
    }
    return 0;
}

torch::Device Device::ToTorchDevice() const noexcept { return external::torch::TorchUtil::ToDevice(*this); }

std::string Device::ToString() const {
    std::stringstream ss;
    ss << DeviceKindToString(deviceKind) << ":" << deviceId;
    return ss.str();
}

bool Device::operator==(const Device& other) const noexcept {
    return deviceKind == other.deviceKind && deviceId == other.deviceId;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
