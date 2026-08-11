/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <unordered_map>

#include "kernel_stream.h"

namespace dtorch {
namespace core {

class KernelStreamManager {
public:
    KernelStreamManager();

    ~KernelStreamManager() = default;

    KernelStream* GetStream(const Device& globalDevice, const Device& localDevice, KernelStreamType streamType,
                            bool isAsync);

    std::vector<KernelStream*> GetAllStream() const noexcept;

    void Sync();

    std::vector<std::pair<int64_t, KernelStreamType>> GetAllCudaStreamId() const noexcept;

private:
    template <typename StreamClass>
    void RegisterStreamConstructor();

    std::unique_ptr<KernelStream> NewStream(KernelStreamKey streamKey, bool isAsync);

private:
    using StreamConstructorFunc =
        std::function<std::unique_ptr<KernelStream>(int64_t deviceId, KernelStreamType streamType, bool isAsync)>;

    std::unordered_map<DeviceKind, StreamConstructorFunc> mStreamConstructorMap;
    StreamKeyMap<std::unique_ptr<KernelStream>> mStreamMap;
    StreamKeyMap<Device> mStreamLocalDeviceMap;
};

}  // namespace core
}  // namespace dtorch
