/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "runner_supported_devices.h"

#include "dtorch/common/debug.h"
#include "dtorch/external/boost/boost_serialization.h"

namespace dtorch {
namespace core {

RunnerSupportedDevices::RunnerSupportedDevices(const DevicePair& device, bool supportCpu)
    : RunnerSupportedDevices(std::vector<DevicePair>({device}), supportCpu) {}

RunnerSupportedDevices::RunnerSupportedDevices(const std::vector<DevicePair>& devices, bool supportCpu)
    : mSupportCpu(supportCpu), mDevices(devices), mGlobalToLocalMap() {
    for (size_t i = 0; i < mDevices.size(); i++) {
        if (mDevices[i].globalDevice.deviceKind == DeviceKind::kCpu ||
            mDevices[i].localDevice.deviceKind == DeviceKind::kCpu) {
            DUnsupportedImpl();
        }

        DeviceKey globalDevice = DeviceKey::FromDevice(mDevices[i].globalDevice);
        DAlwaysAssert(mGlobalToLocalMap.count(globalDevice) == 0);
        mGlobalToLocalMap[globalDevice] = i;
    }

    DAlwaysAssert(mGlobalToLocalMap.size() == mDevices.size());
}

bool RunnerSupportedDevices::IsSupported(const DeviceKeySet& deviceKeySet) {
    for (const auto& device : deviceKeySet) {
        if (mGlobalToLocalMap.count(device) > 0) {
            return true;
        }
        if (mSupportCpu && device.deviceKind == DeviceKind::kCpu) {
            return true;
        }
    }

    return false;
}

StreamKeySet RunnerSupportedDevices::GetSupported(const StreamKeySet& streamKeySet) {
    StreamKeySet result;
    for (const auto& streamKey : streamKeySet) {
        if (mGlobalToLocalMap.count(streamKey.device) > 0) {
            result.insert(streamKey);
        } else if (mSupportCpu && streamKey.device.deviceKind == DeviceKind::kCpu) {
            result.insert(streamKey);
        }
    }
    return result;
}

Device RunnerSupportedDevices::GlobalToLocal(const Device& device) const noexcept {
    if (device.deviceKind == DeviceKind::kCpu) {
        DDebugAssert(mSupportCpu);
        return Device(DeviceKind::kCpu, 0);
    }

    DeviceKey deviceKey = DeviceKey::FromDevice(device);
    auto it = mGlobalToLocalMap.find(deviceKey);
    DDebugAssert(it != mGlobalToLocalMap.end());
    size_t idx = it->second;
    DDebugAssert(mDevices.size() > idx);
    return mDevices[idx].localDevice;
}

std::string RunnerSupportedDevices::SerializeToString() const {
    std::ostringstream oss;
    boost::archive::text_oarchive oa(oss);
    oa << *this;
    return oss.str();
}

RunnerSupportedDevices RunnerSupportedDevices::FromString(const std::string& serializedData) {
    std::istringstream iss(serializedData);
    boost::archive::text_iarchive ia(iss);
    RunnerSupportedDevices devices;
    ia >> devices;
    return devices;
}

}  // namespace core
}  // namespace dtorch
