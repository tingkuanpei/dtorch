/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "file_tensor_store.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "dtorch/common/config.h"
#if DTORCH_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/type_cast.h"
#include "dtorch/core/kernel_stream/kernel_stream.h"
#include "dtorch/external/boost/boost_interprocess.h"
#include "dtorch/external/device/cuda_device_event.h"
#include "dtorch/external/device/device_event.h"
#include "dtorch/external/torch/torch_util.h"

using dtorch::external::device::CudaDeviceEvent;
using dtorch::external::device::DeviceEvent;
using dtorch::external::device::DeviceEventCreateFlag;

namespace dtorch {
namespace core {
namespace communication {

struct SameProcessStore {
    SameProcessStore() : tensorMap(), setEventMap(), getEventMap() {}

    ~SameProcessStore() = default;

    DTORCH_API_DISABLE_COPY_AND_MOVE(SameProcessStore);

public:
    std::unordered_map<std::string, torch::Tensor> tensorMap;
    std::unordered_map<std::string, std::shared_ptr<DeviceEvent>> setEventMap;
    std::unordered_map<std::string, std::vector<std::shared_ptr<DeviceEvent>>> getEventMap;
};

class SameProcessStoreManager {
public:
    DTORCH_FORCEINLINE static SameProcessStoreManager& GetSingleton() {
        static SameProcessStoreManager manager;
        return manager;
    }

public:
    DTORCH_FORCEINLINE std::shared_ptr<SameProcessStore> GetImpl(const std::string& storeKey) {
        std::unique_lock<std::mutex> lock(mMutex);
        CleanExpiredEntries();

        auto it = mImplMap.find(storeKey);
        if (it != mImplMap.end()) {
            std::shared_ptr<SameProcessStore> ptr = it->second.lock();
            DAlwaysAssert(ptr);
            if (ptr) {
                return ptr;
            }
            mImplMap.erase(it);
        }

        std::shared_ptr<SameProcessStore> newInstance = std::make_shared<SameProcessStore>();
        mImplMap[storeKey] = newInstance;
        return newInstance;
    }

private:
    SameProcessStoreManager() : mMutex(), mImplMap() {}

    ~SameProcessStoreManager() = default;

    DTORCH_API_DISABLE_COPY_AND_MOVE(SameProcessStoreManager);

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
    std::unordered_map<std::string, std::weak_ptr<SameProcessStore>> mImplMap;
};

struct FileTensorStore::Impl {
    Impl(const std::string& fileName, int worldSize)
        : fileName(fileName),
          sharedMemory(::boost::interprocess::open_or_create, fileName, GetMemRequireSize(worldSize)),
          stringAllocator(sharedMemory.mMemory->get_segment_manager()),
          stringIntAllocator(sharedMemory.mMemory->get_segment_manager()),
          stringDeviceKindAllocator(sharedMemory.mMemory->get_segment_manager()),
          worldSize(*sharedMemory.FindOrConstruct<int>("WorldSize", worldSize)),
          barrierCounts(*sharedMemory.FindOrConstruct<int>("BarrierCounts", 0)),
          barrierRound(*sharedMemory.FindOrConstruct<int>("BarrierRound", 0)),
          setTensorReadySet(*sharedMemory.FindOrConstructStringSet("SetTensorReady", stringAllocator)),
          expectDestGetDeviceKindMap(
              *sharedMemory.FindOrConstructStringDeviceKindMap("SetDeviceKind", stringDeviceKindAllocator)),
          targetGetCountMap(*sharedMemory.FindOrConstructStringIntMap("TargetGetCount", stringIntAllocator)),
          actualGetCountMap(*sharedMemory.FindOrConstructStringIntMap("ActualGetCount", stringIntAllocator)),
          mutex(*sharedMemory.FindOrConstruct<external::boost::InterprocessMutex>("Mutex")),
          cv(*sharedMemory.FindOrConstruct<external::boost::InterprocessCondition>("Cond")),
          refCount(*sharedMemory.FindOrConstruct<int>("RefCount", 0)),
          sameProcessStorePtr(SameProcessStoreManager::GetSingleton().GetImpl(fileName + "_same_process_store")),
          setEventMap(),
          setEventStreamMap(),
          getEventMap(),
          longStringAutoRemoveMap(),
          longStringMemoryMap() {
        DAlwaysAssert(worldSize > 0);
        {
            external::boost::ScopedLock<external::boost::InterprocessMutex> lock(mutex);
            refCount++;
        }
        // DLogError() << "sharedMemory.mMemory->get_free_memory(): " << sharedMemory.mMemory->get_free_memory();
        // DLogError() << "sharedMemory.mMemory->get_size(): " << sharedMemory.mMemory->get_size();
    }

    ~Impl() {
        bool shouldRemove = false;
        {
            external::boost::ScopedLock<external::boost::InterprocessMutex> lock(mutex);
            DDebugAssert(refCount > 0);
            refCount--;
            shouldRemove = (refCount == 0);
        }
        if (shouldRemove) {
            // Delete interprocess shared memory file when last user exits.
            ::boost::interprocess::shared_memory_object::remove(fileName.c_str());
        }
    }

    DTORCH_API_DISABLE_COPY_AND_MOVE(Impl);

    size_t GetMemRequireSize(int worldSize) {
        DDebugAssert(worldSize > 0);
        size_t requireSize = 20480 + worldSize * 512;
        requireSize = requireSize > 1e7 ? 1e7 : requireSize;
        return requireSize;
    }

public:
    std::string fileName;
    external::boost::ManagedSharedMemory sharedMemory;
    external::boost::ShmStringAllocator stringAllocator;
    external::boost::ShmStringIntAllocator stringIntAllocator;
    external::boost::ShmStringDeviceKindAllocator stringDeviceKindAllocator;
    int& worldSize;
    int& barrierCounts;
    int& barrierRound;
    external::boost::ShmStringSet& setTensorReadySet;
    external::boost::ShmStringDeviceKindMap& expectDestGetDeviceKindMap;
    external::boost::ShmStringIntMap& targetGetCountMap;
    external::boost::ShmStringIntMap& actualGetCountMap;
    external::boost::InterprocessMutex& mutex;
    external::boost::InterprocessCondition& cv;
    int& refCount;

    // Not shared between processes
    // Keep objects alive to avoid race condition
    std::shared_ptr<SameProcessStore> sameProcessStorePtr;
    std::unordered_map<std::string, std::shared_ptr<DeviceEvent>> setEventMap;
    std::unordered_map<std::string, DeviceStream> setEventStreamMap;
    std::unordered_map<std::string, std::vector<std::shared_ptr<DeviceEvent>>> getEventMap;
    std::unordered_map<std::string, std::shared_ptr<external::boost::ShmAutoRemove>> longStringAutoRemoveMap;
    std::unordered_map<std::string, std::shared_ptr<external::boost::ManagedSharedMemory>> longStringMemoryMap;
};

FileTensorStore::FileTensorStore(const std::string& fileName, int worldSize)
    : TensorStore(),
      mResetCount(0),
      mResetBaseFileName(fileName),
      mImplPtr(std::make_shared<Impl>(fileName, worldSize)) {
    Barrier();
}

FileTensorStore::~FileTensorStore() = default;

void FileTensorStore::SrcSet(const std::string& key, const torch::Tensor& value, DeviceStream& stream, size_t getCount,
                             std::optional<DeviceKind> destGetDeviceKind) {
    Device srcValueDevice = external::torch::TorchUtil::GetDevice(value);
    DDebugAssert(srcValueDevice == stream.GetLocalDevice());
    DeviceKind srcValueDeviceKind = srcValueDevice.deviceKind;
    torch::Tensor actualValue = value;
    DeviceStream actualStream = stream;
    DeviceKind actualDestGetDeviceKind = srcValueDeviceKind;
    if (destGetDeviceKind.has_value() && destGetDeviceKind.value() != srcValueDeviceKind) {
        actualDestGetDeviceKind = destGetDeviceKind.value();
        // If value is on CPU and destGetDeviceKind is on GPU, convert value to GPU tensor then use cuda IPC to transfer
        // for better performance.
        if (srcValueDeviceKind == DeviceKind::kCpu) {
            Device destDevice(destGetDeviceKind.value());
            actualValue = value.to(external::torch::TorchUtil::ToDevice(destDevice));
            actualStream = DeviceStream::GetCurrentStream(destDevice);
        }
    }

    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(mImplPtr->mutex);
    external::boost::ShmString shmKey(key.c_str(), mImplPtr->stringAllocator);
    DAlwaysAssert(mImplPtr->setTensorReadySet.find(shmKey) == mImplPtr->setTensorReadySet.end());
    mImplPtr->setEventStreamMap[key] = actualStream;
    mImplPtr->expectDestGetDeviceKindMap.insert(
        std::pair<external::boost::ShmString, DeviceKind>(shmKey, actualDestGetDeviceKind));

    // Current process set tensor
    mImplPtr->sameProcessStorePtr->tensorMap[key] = actualValue;
    mImplPtr->sameProcessStorePtr->setEventMap[key] = DeviceEvent::CreateDeviceEvent(actualStream.GetDeviceKind());
    mImplPtr->sameProcessStorePtr->setEventMap[key]->Record(actualStream);
    mImplPtr->sameProcessStorePtr->setEventMap[key]->SetNvtxName("FileTensorStoreSameProcessStoreSrcSetEvent(" + key +
                                                                 ")");
    mImplPtr->sameProcessStorePtr->getEventMap[key] = std::vector<std::shared_ptr<DeviceEvent>>();

    // Other processes set tensor
    DeviceKind deviceKind = external::torch::TorchUtil::GetDevice(actualValue).deviceKind;
    DDebugAssert(deviceKind == actualStream.GetDeviceKind());
    mImplPtr->sharedMemory.Construct<DeviceKind>(key + "_set_device_kind", deviceKind);
    if (deviceKind == DeviceKind::kGpu) {
        std::string ipcMemHandleStr = external::torch::TorchUtil::ToIpcMemHandle(actualValue);
        std::unique_ptr<DeviceEvent> event =
            DeviceEvent::CreateDeviceEvent(actualStream.GetDeviceKind(), DeviceEventCreateFlag::kInterprocess);
        event->Record(actualStream);
        event->SetNvtxName("FileTensorStoreOtherProcessesSrcSetEvent(" + key + ")");
        std::string ipcEventHandle =
            DerivedCast<CudaDeviceEvent, DeviceEvent>(event.get())->GetNative().GetIpcEventHandle();
        mImplPtr->setEventMap[key] = std::move(event);

        mImplPtr->sharedMemory.ConstructString(key + "_mem", ipcMemHandleStr);
        mImplPtr->sharedMemory.ConstructString(key + "_set_event", ipcEventHandle);
    } else {
        // Cpu tensor all in same process now, will set in mImplPtr->sameProcessStorePtr->tensorMap.

        // if (actualValue.numel() > 1024 * 512) {
        //     DLogWarning() << "Cpu tensor size is too large, will use long time for IPC or RPC transfer.";
        // }

        // std::string ipcMemHandleStr = external::torch::TorchUtil::ToIpcMemHandle(actualValue);
        // SetLongString(key + "_mem", ipcMemHandleStr);
    }
    mImplPtr->targetGetCountMap.insert(
        std::pair<external::boost::ShmString, int64_t>(shmKey, static_cast<int64_t>(getCount)));
    mImplPtr->actualGetCountMap.insert(std::pair<external::boost::ShmString, int64_t>(shmKey, 0));

    mImplPtr->setTensorReadySet.insert(shmKey);

    lock.unlock();
    mImplPtr->cv.notify_all();
}

void FileTensorStore::SrcWaitUntilGetFinished(const std::string& key, DeviceStream& /*stream*/) {
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(mImplPtr->mutex);
    external::boost::ShmString shmKey(key.c_str(), mImplPtr->stringAllocator);

    if (mImplPtr->actualGetCountMap[shmKey] < mImplPtr->targetGetCountMap[shmKey]) {
        mImplPtr->cv.wait(lock,
                          [&] { return mImplPtr->actualGetCountMap[shmKey] == mImplPtr->targetGetCountMap[shmKey]; });
    }
    DDebugAssert(mImplPtr->actualGetCountMap[shmKey] == mImplPtr->targetGetCountMap[shmKey]);

    DDebugAssert(mImplPtr->setEventStreamMap.find(key) != mImplPtr->setEventStreamMap.end());
    DeviceStream actualSrcStream = mImplPtr->setEventStreamMap[key];

    // Current process wait until get finished
    for (auto& event : mImplPtr->sameProcessStorePtr->getEventMap[key]) {
        event->StreamWaitEvent(actualSrcStream);
    }

    // Other processes wait until get finished
    if (mImplPtr->sharedMemory.Count<DeviceKind>(key + "_get_device_kind") > 0) {
        DeviceKind deviceKind = *mImplPtr->sharedMemory.Find<DeviceKind>(key + "_get_device_kind");
        if (deviceKind == DeviceKind::kGpu) {
            for (int64_t i = 0; i < mImplPtr->targetGetCountMap[shmKey]; i++) {
                const std::string& getEventKey = key + "_get_event_" + std::to_string(i);
                if (mImplPtr->sharedMemory.Count<std::string>(getEventKey) == 0) {
                    continue;
                }

                std::string ipcEventHandle = mImplPtr->sharedMemory.FindStr(getEventKey);
                CudaDeviceEvent cudaEvent(ipcEventHandle);
                cudaEvent.SetNvtxName("FileTensorStoreOtherProcessesDestFinishGetEventFromIpc(" + key + ")");
                cudaEvent.StreamWaitEvent(actualSrcStream);
            }
        }
    }

    // keep source tensor storage alive until producer stream observes all get-finished events
#if DTORCH_WITH_CUDA
    DDebugAssert(mImplPtr->sameProcessStorePtr->tensorMap.find(key) != mImplPtr->sameProcessStorePtr->tensorMap.end());
    auto actualSrcTensor = mImplPtr->sameProcessStorePtr->tensorMap[key];
    if (external::torch::TorchUtil::GetDevice(actualSrcTensor).deviceKind == DeviceKind::kGpu &&
        actualSrcStream.GetLocalDevice().deviceKind == DeviceKind::kGpu) {
        c10::cuda::CUDACachingAllocator::recordStream(actualSrcTensor.storage().data_ptr(),
                                                      *(actualSrcStream.GetTorchCudaStream()));
    }
#endif
}

torch::Tensor FileTensorStore::DestGet(const std::string& key, DeviceStream& stream) {
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(mImplPtr->mutex);
    external::boost::ShmString shmKey(key.c_str(), mImplPtr->stringAllocator);

    if (mImplPtr->setTensorReadySet.find(shmKey) == mImplPtr->setTensorReadySet.end()) {
        mImplPtr->cv.wait(
            lock, [&] { return mImplPtr->setTensorReadySet.find(shmKey) != mImplPtr->setTensorReadySet.end(); });
    }
    DDebugAssert(mImplPtr->setTensorReadySet.find(shmKey) != mImplPtr->setTensorReadySet.end());

    torch::Tensor srcTensor;
    bool needClone = false;
    if (mImplPtr->sameProcessStorePtr->tensorMap.count(key) > 0) {
        // Current process get tensor
        srcTensor = mImplPtr->sameProcessStorePtr->tensorMap[key];
        mImplPtr->sameProcessStorePtr->setEventMap[key]->StreamWaitEvent(stream);
    } else {
        // Other processes get tensor
        DeviceKind deviceKind = *mImplPtr->sharedMemory.Find<DeviceKind>(key + "_set_device_kind");
        if (deviceKind == DeviceKind::kGpu) {
            std::string ipcMemHandleStr = mImplPtr->sharedMemory.FindStr(key + "_mem");
            std::string ipcEventHandle = mImplPtr->sharedMemory.FindStr(key + "_set_event");

            torch::Tensor tensor = external::torch::TorchUtil::FromIpcMemHandle(ipcMemHandleStr);
            CudaDeviceEvent cudaEvent(ipcEventHandle);
            cudaEvent.SetNvtxName("FileTensorStoreOtherProcessesSrcSetEventFromIpc(" + key + ")");
            cudaEvent.StreamWaitEvent(stream);
            // Imported CUDA IPC memory cannot be exported again with cudaIpcGetMemHandle.
            // Clone here to materialize storage owned by the current process before any later SrcSet().
            needClone = true;
            srcTensor = tensor;
        } else {
            // Cpu tensor all in same process now, will get from mImplPtr->sameProcessStorePtr->tensorMap.
            DUnsupportedImpl();
            // std::string ipcMemHandleStr = GetLongString(key + "_mem");
            // torch::Tensor tensor = external::torch::TorchUtil::FromIpcMemHandle(ipcMemHandleStr);
            // srcTensor = tensor;
        }
    }

    // Each thread has different src cuda stream. Operator for tensor may happen in src cuda stream in this thread.
    auto event = external::device::DeviceEvent::CreateDeviceEvent(stream.GetDeviceKind());
    event->Record(stream);
    DeviceStream srcDeviceStreamOfThisThread =
        DeviceStream::GetCurrentStream(external::torch::TorchUtil::GetDevice(srcTensor));
    event->StreamWaitEvent(srcDeviceStreamOfThisThread);

    DeviceKind actualDestGetDeviceKind = mImplPtr->expectDestGetDeviceKindMap[shmKey];
    if (actualDestGetDeviceKind != external::torch::TorchUtil::GetDevice(srcTensor).deviceKind) {
        srcTensor = srcTensor.to(external::torch::TorchUtil::ToDevice(actualDestGetDeviceKind));
    }

    return needClone ? srcTensor.clone() : srcTensor;
}

void FileTensorStore::DestFinishGet(const std::string& key, DeviceStream& stream) {
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(mImplPtr->mutex);
    external::boost::ShmString shmKey(key.c_str(), mImplPtr->stringAllocator);
    DDebugAssert(mImplPtr->actualGetCountMap[shmKey] < mImplPtr->targetGetCountMap[shmKey]);

    if (mImplPtr->sameProcessStorePtr->tensorMap.count(key) > 0) {
        // Current process finish get tensor
        auto event = DeviceEvent::CreateDeviceEvent(stream.GetDeviceKind());
        event->Record(stream);
        event->SetNvtxName("FileTensorStoreSameProcessStoreDestFinishGetEvent(" + key + ")");
        mImplPtr->sameProcessStorePtr->getEventMap[key].push_back(std::move(event));
    } else {
        // Other processes finish get tensor
        DeviceKind deviceKind = stream.GetDeviceKind();
        if (mImplPtr->sharedMemory.Count<DeviceKind>(key + "_get_device_kind") > 0) {
            DAlwaysAssert(deviceKind == *mImplPtr->sharedMemory.Find<DeviceKind>(key + "_get_device_kind"));
        } else {
            mImplPtr->sharedMemory.Construct<DeviceKind>(key + "_get_device_kind", deviceKind);
        }

        if (deviceKind == DeviceKind::kGpu) {
            int64_t& actualGetCount = mImplPtr->actualGetCountMap[shmKey];
            std::unique_ptr<DeviceEvent> event =
                DeviceEvent::CreateDeviceEvent(stream.GetDeviceKind(), DeviceEventCreateFlag::kInterprocess);
            event->Record(stream);
            event->SetNvtxName("FileTensorStoreOtherProcessesDestFinishGetEvent(" + key + ")");
            std::string ipcEventHandle =
                DerivedCast<CudaDeviceEvent, DeviceEvent>(event.get())->GetNative().GetIpcEventHandle();
            mImplPtr->getEventMap[key].push_back(std::move(event));
            mImplPtr->sharedMemory.ConstructString(key + "_get_event_" + std::to_string(actualGetCount),
                                                   ipcEventHandle);
        }
    }

    mImplPtr->actualGetCountMap[shmKey] = mImplPtr->actualGetCountMap[shmKey] + 1;
    if (mImplPtr->actualGetCountMap[shmKey] == mImplPtr->targetGetCountMap[shmKey]) {
        lock.unlock();
        mImplPtr->cv.notify_all();
    }
}

void FileTensorStore::SetLongString(const std::string& key, const std::string& value) {
    size_t requireSize = value.size() / 1024 * 1024 + 4096;

    std::string longStrFileName = mImplPtr->fileName + "_" + key + "_long_string";
    DDebugAssert(mImplPtr->longStringAutoRemoveMap.find(key) == mImplPtr->longStringAutoRemoveMap.end());
    DDebugAssert(mImplPtr->longStringMemoryMap.find(key) == mImplPtr->longStringMemoryMap.end());
    mImplPtr->longStringAutoRemoveMap[key] = std::make_shared<external::boost::ShmAutoRemove>(longStrFileName);
    std::shared_ptr<external::boost::ManagedSharedMemory> newMemory =
        std::make_shared<external::boost::ManagedSharedMemory>(::boost::interprocess::create_only,
                                                               longStrFileName.c_str(), requireSize);
    external::boost::ShmStringAllocator stringAllocator(newMemory->mMemory->get_segment_manager());
    newMemory->ConstructString(key.c_str(), value);
    mImplPtr->longStringMemoryMap[key] = newMemory;
}

std::string FileTensorStore::GetLongString(const std::string& key) const {
    std::string longStrFileName = mImplPtr->fileName + "_" + key + "_long_string";
    external::boost::ManagedSharedMemory sharedMemory(::boost::interprocess::open_only, longStrFileName.c_str());
    std::string result = sharedMemory.FindStr(key);
    return result;
}

void FileTensorStore::Reset() {
    Barrier();

    mResetCount++;
    std::string fileName = mResetBaseFileName + "_Count_" + std::to_string(mResetCount);
    mImplPtr = std::make_shared<Impl>(fileName, mImplPtr->worldSize);

    Barrier();
}

void FileTensorStore::Barrier() {
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(mImplPtr->mutex);
    DDebugAssert(0 <= mImplPtr->barrierCounts && mImplPtr->barrierCounts < mImplPtr->worldSize);
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
