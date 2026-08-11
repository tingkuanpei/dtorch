/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "memory_tensor_store.h"

#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include "dtorch/common/config.h"
#if DTORCH_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "dtorch/core/kernel_stream/kernel_stream.h"
#include "dtorch/external/device/cuda_device_event.h"
#include "dtorch/external/device/device_event.h"

using dtorch::external::device::DeviceEvent;

namespace dtorch {
namespace core {
namespace communication {

struct MemoryTensorStore::Impl {
    Impl()
        : mutex(),
          cv(),
          tensorMap(),
          setEventMap(),
          setEventStreamMap(),
          targetGetCountMap(),
          actualGetCountMap(),
          worldSize(0),
          barrierCounts(0),
          barrierRound(0) {}

    ~Impl() = default;

    DTORCH_API_DISABLE_COPY_AND_MOVE(Impl);

public:
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<std::string, torch::Tensor> tensorMap;
    std::unordered_map<std::string, std::shared_ptr<DeviceEvent>> setEventMap;
    std::unordered_map<std::string, DeviceStream> setEventStreamMap;
    std::unordered_map<std::string, size_t> targetGetCountMap;
    std::unordered_map<std::string, size_t> actualGetCountMap;
    int worldSize;
    int barrierCounts;
    int barrierRound;
};

class MemoryTensorStoreImpManager {
public:
    DTORCH_FORCEINLINE static MemoryTensorStoreImpManager& GetSingleton() {
        static MemoryTensorStoreImpManager manager;
        return manager;
    }

public:
    DTORCH_FORCEINLINE std::shared_ptr<MemoryTensorStore::Impl> GetImpl(const std::string& storeKey) {
        std::unique_lock<std::mutex> lock(mMutex);
        CleanExpiredEntries();

        auto it = mImplMap.find(storeKey);
        if (it != mImplMap.end()) {
            std::shared_ptr<MemoryTensorStore::Impl> ptr = it->second.lock();
            DAlwaysAssert(ptr);
            return ptr;
        }

        std::shared_ptr<MemoryTensorStore::Impl> newInstance = std::make_shared<MemoryTensorStore::Impl>();
        mImplMap[storeKey] = newInstance;
        return newInstance;
    }

private:
    MemoryTensorStoreImpManager() : mMutex(), mImplMap() {}

    ~MemoryTensorStoreImpManager() = default;

    DTORCH_API_DISABLE_COPY_AND_MOVE(MemoryTensorStoreImpManager);

    DTORCH_FORCEINLINE void CleanExpiredEntries() {
        for (auto it = mImplMap.begin(); it != mImplMap.end();) {
            if (it->second.expired()) {
                it = mImplMap.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    std::mutex mMutex;
    std::unordered_map<std::string, std::weak_ptr<MemoryTensorStore::Impl>> mImplMap;
};

MemoryTensorStore::MemoryTensorStore(const std::string& storeKey, int worldSize)
    : mImplPtr(MemoryTensorStoreImpManager::GetSingleton().GetImpl(storeKey)) {
    std::unique_lock<std::mutex> lock(mImplPtr->mutex);
    DAlwaysAssert(worldSize > 0);
    if (mImplPtr->worldSize == 0) {
        mImplPtr->worldSize = worldSize;
    } else {
        DAlwaysAssert(mImplPtr->worldSize == worldSize);
    }
    lock.unlock();

    Barrier();
}

MemoryTensorStore::~MemoryTensorStore() = default;

void MemoryTensorStore::SrcSet(const std::string& key, const torch::Tensor& value, DeviceStream& stream,
                               size_t getCount, std::optional<DeviceKind> destGetDeviceKind) {
    Device srcValueDevice = external::torch::TorchUtil::GetDevice(value);
    DAlwaysAssert(srcValueDevice == stream.GetLocalDevice());
    DeviceKind srcValueDeviceKind = srcValueDevice.deviceKind;
    torch::Tensor actualValue = value;
    DeviceStream actualStream = stream;
    if (destGetDeviceKind.has_value() && srcValueDeviceKind != destGetDeviceKind.value()) {
        Device destDevice(destGetDeviceKind.value());
        actualValue = value.to(external::torch::TorchUtil::ToDevice(destDevice));
        actualStream = DeviceStream::GetCurrentStream(destDevice);
    }

    std::unique_lock<std::mutex> lock(mImplPtr->mutex);
    DAlwaysAssert(mImplPtr->tensorMap.find(key) == mImplPtr->tensorMap.end());
    mImplPtr->tensorMap[key] = actualValue;
    mImplPtr->setEventMap[key] = DeviceEvent::CreateDeviceEvent(actualStream.GetDeviceKind());
    mImplPtr->setEventMap[key]->Record(actualStream);
    mImplPtr->setEventMap[key]->SetNvtxName("MemoryTensorStoreSrcSetEvent(" + key + ")");
    mImplPtr->setEventStreamMap[key] = actualStream;
    mImplPtr->targetGetCountMap[key] = getCount;
    mImplPtr->actualGetCountMap[key] = 0;
    lock.unlock();
    mImplPtr->cv.notify_all();
}

void MemoryTensorStore::SrcWaitUntilGetFinished(const std::string& key, DeviceStream& /*stream*/) {
    std::unique_lock<std::mutex> lock(mImplPtr->mutex);

    if (mImplPtr->actualGetCountMap.at(key) < mImplPtr->targetGetCountMap.at(key)) {
        mImplPtr->cv.wait(lock,
                          [&] { return mImplPtr->actualGetCountMap.at(key) == mImplPtr->targetGetCountMap.at(key); });
    }
    DAlwaysAssert(mImplPtr->actualGetCountMap.at(key) == mImplPtr->targetGetCountMap.at(key));
    DAlwaysAssert(mImplPtr->setEventStreamMap.find(key) != mImplPtr->setEventStreamMap.end());
}

torch::Tensor MemoryTensorStore::DestGet(const std::string& key, DeviceStream& stream) {
    std::unique_lock<std::mutex> lock(mImplPtr->mutex);
    if (mImplPtr->tensorMap.find(key) == mImplPtr->tensorMap.end()) {
        mImplPtr->cv.wait(lock, [&] { return mImplPtr->tensorMap.find(key) != mImplPtr->tensorMap.end(); });
    }
    DAlwaysAssert(mImplPtr->tensorMap.find(key) != mImplPtr->tensorMap.end());
    mImplPtr->setEventMap[key]->StreamWaitEvent(stream);
    return mImplPtr->tensorMap[key];
}

void MemoryTensorStore::DestFinishGet(const std::string& key, DeviceStream& stream) {
    std::unique_lock<std::mutex> lock(mImplPtr->mutex);
    DAlwaysAssert(mImplPtr->actualGetCountMap.at(key) < mImplPtr->targetGetCountMap.at(key));

    // keep source tensor storage alive until producer stream observes all get-finished events
#if DTORCH_WITH_CUDA
    DAlwaysAssert(mImplPtr->tensorMap.find(key) != mImplPtr->tensorMap.end());
    auto srcTensor = mImplPtr->tensorMap[key];
    if (external::torch::TorchUtil::GetDevice(srcTensor).deviceKind == DeviceKind::kGpu &&
        stream.GetLocalDevice().deviceKind == DeviceKind::kGpu) {
        c10::cuda::CUDACachingAllocator::recordStream(srcTensor.storage().data_ptr(), *(stream.GetTorchCudaStream()));
    }
#endif

    mImplPtr->actualGetCountMap.at(key) = mImplPtr->actualGetCountMap.at(key) + 1;
    if (mImplPtr->actualGetCountMap.at(key) == mImplPtr->targetGetCountMap.at(key)) {
        lock.unlock();
        mImplPtr->cv.notify_all();
    }
}

void MemoryTensorStore::Reset() {
    Barrier();
    {
        std::unique_lock<std::mutex> lock(mImplPtr->mutex);
        mImplPtr->tensorMap.clear();
        mImplPtr->setEventMap.clear();
        mImplPtr->setEventStreamMap.clear();
        mImplPtr->targetGetCountMap.clear();
        mImplPtr->actualGetCountMap.clear();
    }
    Barrier();
}

void MemoryTensorStore::Barrier() {
    std::unique_lock<std::mutex> lock(mImplPtr->mutex);
    DAlwaysAssert(0 <= mImplPtr->barrierCounts && mImplPtr->barrierCounts < mImplPtr->worldSize);

    int currentRound = mImplPtr->barrierRound;
    mImplPtr->barrierCounts++;

    if (mImplPtr->barrierCounts == mImplPtr->worldSize) {
        mImplPtr->barrierCounts = 0;
        mImplPtr->barrierRound++;
        lock.unlock();
        mImplPtr->cv.notify_all();
    } else {
        mImplPtr->cv.wait(lock, [this, currentRound]() { return currentRound != mImplPtr->barrierRound; });
    }
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
