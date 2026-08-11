/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <sstream>
#include <unordered_map>

#include "dtorch/common/string.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

struct DevicePair {
    Device globalDevice;
    Device localDevice;

public:
    DevicePair() : globalDevice(), localDevice() {}

    DevicePair(Device globalDevice, Device localDevice) : globalDevice(globalDevice), localDevice(localDevice) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & globalDevice;
        ar & localDevice;
    }

    std::string ToString() const {
        std::stringstream ss;
        ss << "DevicePair(global: " << globalDevice << ", local: " << localDevice << ")";
        return ss.str();
    }

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const DevicePair& pair) {
        os << pair.ToString();
        return os;
    }
};

// Global devices supported by Runner and their corresponding local devices
class RunnerSupportedDevices {
public:
    RunnerSupportedDevices(const DevicePair& device, bool supportCpu = false);

    RunnerSupportedDevices(const std::vector<DevicePair>& devices = {}, bool supportCpu = false);

    DTORCH_FORCEINLINE bool IsSupportCpu() const noexcept { return mSupportCpu; }

    DTORCH_FORCEINLINE const std::vector<DevicePair>& AllDevices() const noexcept { return mDevices; }

    DTORCH_FORCEINLINE size_t DeviceSize() const noexcept { return mDevices.size(); }

    DTORCH_FORCEINLINE std::vector<Device> AllDeviceList() const noexcept {
        std::vector<Device> result;
        for (const auto& pair : mDevices) {
            result.push_back(pair.globalDevice);
        }
        return result;
    }

    bool IsSupported(const DeviceKeySet& deviceKeySet);

    StreamKeySet GetSupported(const StreamKeySet& streamKeySet);

    Device GlobalToLocal(const Device& device) const noexcept;

    // Boost-text-archive (de)serialization for cross-node transfer via gRPC (CreateGraph).
    std::string SerializeToString() const;
    static RunnerSupportedDevices FromString(const std::string& serializedData);

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mSupportCpu;
        ar & mDevices;
        ar & mGlobalToLocalMap;
    }

    std::string ToString() const {
        std::stringstream ss;
        ss << "RunnerSupportedDevices[supportCpu: " << mSupportCpu << ", devices: " << String::ToString(mDevices)
           << "]";
        return ss.str();
    }

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const RunnerSupportedDevices& devices) {
        os << devices.ToString();
        return os;
    }

private:
    bool mSupportCpu;
    std::vector<DevicePair> mDevices;
    DeviceKeyMap<size_t> mGlobalToLocalMap;
};

}  // namespace core
}  // namespace dtorch
