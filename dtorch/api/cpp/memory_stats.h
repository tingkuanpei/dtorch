/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "device.h"

namespace dtorch {
namespace api {
namespace cpp {

struct MemoryStat {
    int64_t allocated;
    int64_t reserved;
    int64_t maxAllocated;
    int64_t maxReserved;

public:
    MemoryStat() : allocated(0), reserved(0), maxAllocated(0), maxReserved(0) {}

    MemoryStat(int64_t allocated, int64_t reserved, int64_t maxAllocated, int64_t maxReserved)
        : allocated(allocated), reserved(reserved), maxAllocated(maxAllocated), maxReserved(maxReserved) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & allocated;
        ar & reserved;
        ar & maxAllocated;
        ar & maxReserved;
    }

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const MemoryStat& memoryStat) {
        os << memoryStat.ToString();
        return os;
    }

    DTORCH_API_FORCEINLINE bool operator==(const MemoryStat& other) const {
        return allocated == other.allocated && reserved == other.reserved && maxAllocated == other.maxAllocated &&
               maxReserved == other.maxReserved;
    }

    DTORCH_API_FORCEINLINE bool operator!=(const MemoryStat& other) const { return !(*this == other); }
};

class MemoryStats {
public:
    using Iterator = DeviceKeyMap<MemoryStat>::iterator;
    using ConstIterator = DeviceKeyMap<MemoryStat>::const_iterator;

public:
    MemoryStats() : memoryStats(), dtensorInSameDevice(std::nullopt) {}

    ~MemoryStats() = default;

    void Merge(const MemoryStats& memoryStats);

    void MergeAsDTensorInSameDevice();

    DTORCH_API_FORCEINLINE size_t Size() const { return memoryStats.size(); }

    DTORCH_API_FORCEINLINE std::pair<Iterator, bool> Insert(const DeviceKeyMap<MemoryStat>::value_type& value) {
        return memoryStats.insert(value);
    }

    DTORCH_API_FORCEINLINE std::pair<Iterator, bool> Insert(DeviceKeyMap<MemoryStat>::value_type&& value) {
        return memoryStats.insert(std::move(value));
    }

    DTORCH_API_FORCEINLINE Iterator Find(int64_t deviceId) {
        return memoryStats.find(DeviceKey::FromDevice(Device(DeviceKind::kGpu, deviceId)));
    }

    DTORCH_API_FORCEINLINE ConstIterator Find(int64_t deviceId) const {
        return memoryStats.find(DeviceKey::FromDevice(Device(DeviceKind::kGpu, deviceId)));
    }

    DTORCH_API_FORCEINLINE Iterator Begin() { return memoryStats.begin(); }

    DTORCH_API_FORCEINLINE ConstIterator Begin() const { return memoryStats.begin(); }

    DTORCH_API_FORCEINLINE Iterator End() { return memoryStats.end(); }

    DTORCH_API_FORCEINLINE ConstIterator End() const { return memoryStats.end(); }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & memoryStats;
        ar & dtensorInSameDevice;
    }

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const MemoryStats& memoryStats) {
        os << memoryStats.ToString();
        return os;
    }

private:
    DeviceKeyMap<MemoryStat> memoryStats;
    std::optional<MemoryStat> dtensorInSameDevice;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
