/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "api_type.h"
#include "api_utilities.h"
#include "params_hash.h"
#include "serialization.h"

namespace dtorch {
namespace api {
namespace cpp {

enum class DeviceKind : int64_t {
    kCpu = 0,
    kGpu,
    kCount,
};

DeviceKind DeviceKindFromString(const std::string& deviceStr);

std::string DeviceKindToString(DeviceKind deviceKind);

std::ostream& operator<<(std::ostream& os, DeviceKind deviceKind);

struct Device {
    DeviceKind deviceKind;
    int64_t deviceId;

public:
    static const Device& GetDefaultCpuDevice() noexcept {
        static Device device(DeviceKind::kCpu);
        return device;
    }

    static bool IsAvailable(DeviceKind deviceKind) noexcept;

    static size_t DeviceCount(DeviceKind deviceKind) noexcept;

public:
    Device(DeviceKind deviceKind = DeviceKind::kCpu, int64_t deviceId = 0) noexcept
        : deviceKind(deviceKind), deviceId(deviceId) {}

    Device(torch::Device torchDevice);

    Device(const std::string& deviceStr);

    bool operator==(const Device& other) const noexcept;
    DTORCH_API_FORCEINLINE bool operator!=(const Device& other) const noexcept { return !(this->operator==(other)); }

    torch::Device ToTorchDevice() const noexcept;

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const Device& device) {
        os << device.ToString();
        return os;
    }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & deviceKind;
        ar & deviceId;
    }
};

// DeviceKey
// DeviceKey MUST POD type
struct DeviceKey {
    DeviceKind deviceKind;
    int64_t deviceId;

public:
    DTORCH_API_FORCEINLINE static DeviceKey FromDevice(const Device& device) {
        DeviceKey result;
        // DeviceKey maybe align, set it all to 0
        std::memset(&result, 0, sizeof(DeviceKey));
        result.deviceKind = device.deviceKind;
        result.deviceId = device.deviceId;
        return result;
    }

    DTORCH_API_FORCEINLINE static Device ToDevice(const DeviceKey& device) {
        return Device(device.deviceKind, device.deviceId);
    }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        if constexpr (Archive::is_loading::value) {
            std::memset(this, 0, sizeof(DeviceKey));
        }
        ar & deviceKind;
        ar & deviceId;
    }

    std::string ToString() const {
        std::stringstream ss;
        ss << "DeviceKey(" << DeviceKindToString(deviceKind) << ":" << deviceId << ")";
        return ss.str();
    }

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const DeviceKey& device) {
        os << device.ToString();
        return os;
    }
};

static_assert(sizeof(DeviceKey) == sizeof(DeviceKind) + sizeof(int64_t),
              "DeviceKey MUST POD type without memory align");

template <class Type>
using DeviceKeyMap = std::unordered_map<DeviceKey, Type, ParamsHash<DeviceKey>, ParamsEqual<DeviceKey>>;

using DeviceKeySet = std::unordered_set<DeviceKey, ParamsHash<DeviceKey>, ParamsEqual<DeviceKey>>;

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
