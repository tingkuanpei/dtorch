/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <sstream>
#include <string>

#include "dtorch/api/cpp/api_utilities.h"
#include "dtorch/api/cpp/serialization.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

class GlobalOption {
public:
    DTORCH_API_FORCEINLINE static GlobalOption& GetSingleton() {
        static GlobalOption option;
        return option;
    }

    // Initialize singleton from MainNode's serialized data (WorkerNode side).
    // Must be called before GetSingleton() is first accessed.
    static void GlobalOptionInitFromString(const std::string& serializedData);

public:
    ~GlobalOption() = default;

    DTORCH_API_FORCEINLINE int64_t GetCommTimeoutSecond() const noexcept { return mCommTimeoutSecond; }

    DTORCH_API_FORCEINLINE int64_t GetGrpcTimeoutSecond() const noexcept { return mGrpcTimeoutSecond; }

    DTORCH_API_FORCEINLINE int64_t GetZmqTimeoutSecond() const noexcept { return mZmqTimeoutSecond; }

    DTORCH_API_FORCEINLINE bool GetDTensorInSameDevice() const noexcept { return mDTensorInSameDevice; }

    DTORCH_API_FORCEINLINE bool GetPerDevicePerProcess() const noexcept { return mPerDevicePerProcess; }

    DTORCH_API_FORCEINLINE int64_t GetNumGpuWhenEnableDtensorInSameDevice() const noexcept {
        return mNumGpuWhenEnableDtensorInSameDevice;
    }

    DTORCH_API_FORCEINLINE bool GetValidateKernelInputOutput() const noexcept { return mValidateKernelInputOutput; }

    // Serialize to string for cross-process transfer (MainNode side)
    std::string SerializeToString() const;

    friend Serialization;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mCommTimeoutSecond;
        ar & mGrpcTimeoutSecond;
        ar & mZmqTimeoutSecond;
        ar & mDTensorInSameDevice;
        ar & mPerDevicePerProcess;
        ar & mNumGpuWhenEnableDtensorInSameDevice;
        ar & mValidateKernelInputOutput;
    }

private:
    GlobalOption();

    // Tag constructor: default-initializes fields without reading env vars (used for deserialization)
    struct DeserializeTag {};
    explicit GlobalOption(DeserializeTag);

    // Copy field values from another GlobalOption
    GlobalOption& operator=(const GlobalOption& other);

private:
    int64_t mCommTimeoutSecond;
    int64_t mGrpcTimeoutSecond;
    int64_t mZmqTimeoutSecond;
    bool mDTensorInSameDevice;
    bool mPerDevicePerProcess;
    int64_t mNumGpuWhenEnableDtensorInSameDevice;
    bool mValidateKernelInputOutput;
};

}  // namespace core
}  // namespace dtorch
