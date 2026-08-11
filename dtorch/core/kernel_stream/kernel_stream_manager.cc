/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "kernel_stream_manager.h"

#include <utility>

#include "cpu_kernel_stream.h"
#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/core/type.h"
#if DTORCH_WITH_CUDA
#include "cuda_kernel_stream.h"
#endif

namespace dtorch {
namespace core {

template <typename StreamClass>
void KernelStreamManager::RegisterStreamConstructor() {
    DeviceKind kind = StreamClass::SupportDeviceKind();

    if (mStreamConstructorMap.count(kind) > 0) {
        DLogFatal() << "Can't register stream constructor at same kind: " << kind << "."
                    << "Please check StreamClass::SupportDeviceKind()";
    }

    auto streamConstructorFunc = [](int64_t deviceId, KernelStreamType streamType,
                                    bool isAsync) -> std::unique_ptr<KernelStream> {
        return std::unique_ptr<KernelStream>(new StreamClass(deviceId, streamType, isAsync));
    };
    mStreamConstructorMap[kind] = streamConstructorFunc;
}

std::unique_ptr<KernelStream> KernelStreamManager::NewStream(KernelStreamKey streamKey, bool isAsync) {
    auto iterator = mStreamConstructorMap.find(streamKey.device.deviceKind);
    if (iterator == mStreamConstructorMap.end()) {
        std::stringstream ss;
        ss << "Unsupported stream: " << streamKey.device.deviceKind << std::endl;
        ss << "Candidate: ";
        for (const auto& it : mStreamConstructorMap) {
            ss << it.first << ", ";
        }
        ss << std::endl;

        DLogFatal() << ss.str();
    }

    return iterator->second(streamKey.device.deviceId, streamKey.streamType, isAsync);
}

KernelStreamManager::KernelStreamManager() : mStreamConstructorMap(), mStreamMap(), mStreamLocalDeviceMap() {
    RegisterStreamConstructor<CpuKernelStream>();
#if DTORCH_WITH_CUDA
    RegisterStreamConstructor<CudaKernelStream>();
#endif
}

KernelStream* KernelStreamManager::GetStream(const Device& globalDevice, const Device& localDevice,
                                             KernelStreamType streamType, bool isAsync) {
    KernelStreamKey searchStreamKey;
    searchStreamKey.Init(globalDevice, streamType);

    auto it = mStreamMap.find(searchStreamKey);
    if (it == mStreamMap.end()) {
        KernelStreamKey createStreamKey;
        createStreamKey.Init(localDevice, streamType);

        DDebugAssert(mStreamLocalDeviceMap.count(searchStreamKey) == 0);
        mStreamLocalDeviceMap[searchStreamKey] = localDevice;

        mStreamMap[searchStreamKey] = NewStream(createStreamKey, isAsync);
        return mStreamMap[searchStreamKey].get();
    } else {
        DDebugAssert(mStreamLocalDeviceMap[searchStreamKey] == localDevice);
    }

    return it->second.get();
}

std::vector<KernelStream*> KernelStreamManager::GetAllStream() const noexcept {
    std::vector<KernelStream*> result;
    for (auto& it : mStreamMap) {
        result.push_back(it.second.get());
    }
    return result;
}

void KernelStreamManager::Sync() {
    for (auto& it : mStreamMap) {
        it.second->Sync();
    }
}

std::vector<std::pair<int64_t, KernelStreamType>> KernelStreamManager::GetAllCudaStreamId() const noexcept {
    std::vector<std::pair<int64_t, KernelStreamType>> result;
    for (const auto& it : mStreamMap) {
        if (it.first.device.deviceKind == DeviceKind::kGpu) {
            result.push_back(std::make_pair(it.first.device.deviceId, it.first.streamType));
        }
    }
    return result;
}

}  // namespace core
}  // namespace dtorch
