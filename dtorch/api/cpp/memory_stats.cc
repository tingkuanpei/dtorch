/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "memory_stats.h"

#include <algorithm>
#include <sstream>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace api {
namespace cpp {

std::string MemoryStat::ToString() const {
    std::stringstream ss;
    ss << "MemoryStat(allocated: " << allocated / 1e6 << "MB, reserved: " << reserved / 1e6 << "MB"
       << ", maxAllocated: " << maxAllocated / 1e6 << "MB, maxReserved: " << maxReserved / 1e6 << "MB"
       << ")";
    return ss.str();
}

void MemoryStats::Merge(const MemoryStats& otherMemoryStats) {
    for (const auto& [device, memoryStat] : otherMemoryStats.memoryStats) {
        DDebugAssert(memoryStats.find(device) == memoryStats.end());
        memoryStats.insert({device, memoryStat});
    }
}

std::string MemoryStats::ToString() const {
    std::vector<DeviceKey> devices;
    for (const auto& it : memoryStats) {
        devices.push_back(it.first);
    }
    std::sort(devices.begin(), devices.end(),
              [](const DeviceKey& a, const DeviceKey& b) { return a.deviceId < b.deviceId; });

    std::stringstream ss;
    ss << "MemoryStats(" << std::endl;
    for (const auto& device : devices) {
        DDebugAssert(memoryStats.find(device) != memoryStats.end());
        ss << "\t" << device << ": " << memoryStats.at(device) << std::endl;
    }
    if (dtensorInSameDevice.has_value()) {
        ss << "\tdtensorInSameDevice: " << dtensorInSameDevice.value() << std::endl;
    }
    ss << ")";
    return ss.str();
}

void MemoryStats::MergeAsDTensorInSameDevice() {
    DDebugAssert(!dtensorInSameDevice.has_value());
    MemoryStat dtensorInSameDeviceMemoryStat;
    for (const auto& [device, memoryStat] : memoryStats) {
        dtensorInSameDeviceMemoryStat.allocated += memoryStat.allocated;
        dtensorInSameDeviceMemoryStat.reserved += memoryStat.reserved;
        dtensorInSameDeviceMemoryStat.maxAllocated += memoryStat.maxAllocated;
        dtensorInSameDeviceMemoryStat.maxReserved += memoryStat.maxReserved;
    }
    dtensorInSameDevice = dtensorInSameDeviceMemoryStat;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
