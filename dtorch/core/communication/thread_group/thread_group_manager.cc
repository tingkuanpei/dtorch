/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "thread_group_manager.h"

#include <utility>

#include "dtorch/common/debug.h"
#include "dtorch/common/filesystem.h"
#include "dtorch/common/logging.h"
#include "dtorch/external/boost/boost_interprocess.h"

namespace dtorch {
namespace core {
namespace communication {

ThreadGroupInfo::ThreadGroupInfo(const SimpleArray& array, size_t dim, int64_t deviceId)
    : mSubGroup(0), mAllDeviceId() {
    auto shape = array.GetShape();
    DDebugAssert(dim < shape.NumAxis());

    if (shape.NumAxis() == 1) {
        mSubGroup = 0;
        mAllDeviceId = array.GetData();
        return;
    }

    size_t leftSize = 1;
    size_t rightSize = 1;
    size_t midSize = shape[dim];
    for (size_t i = 0; i < dim; i++) {
        leftSize *= shape[i];
    }
    for (size_t i = dim + 1; i < shape.NumAxis(); i++) {
        rightSize *= shape[i];
    }

    shape = Shape({leftSize, midSize, rightSize});
    Stride stride(shape);
    const auto& arrayData = array.GetData();
    DDebugAssert(leftSize + rightSize >= 2);
    for (size_t leftIdx = 0; leftIdx < leftSize; leftIdx++) {
        for (size_t rightIdx = 0; rightIdx < rightSize; rightIdx++) {
            bool match = false;
            for (size_t midIdx = 0; midIdx < midSize; midIdx++) {
                size_t idx = stride.ComputeIndex<size_t>({leftIdx, midIdx, rightIdx});
                DDebugAssert(idx < arrayData.size());
                if (arrayData[idx] == deviceId) {
                    match = true;
                    break;
                }
            }

            if (match) {
                mSubGroup = leftIdx * rightSize + rightIdx;
                for (size_t midIdx = 0; midIdx < midSize; midIdx++) {
                    size_t idx = stride.ComputeIndex<size_t>({leftIdx, midIdx, rightIdx});
                    DDebugAssert(idx < arrayData.size());
                    mAllDeviceId.push_back(arrayData[idx]);
                }
                return;
            }
        }
    }

    DUnsupportedImpl();
}

ThreadGroupManager::~ThreadGroupManager() {
    // ncclCommDestroy: This function is an intra-node collective call, which all ranks on the same node should call to
    //                  avoid a hang.
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html
    // We must destroy thread group under fixed order to avoid a hang.

    std::vector<ThreadGroupKey> allThreadGroupKeys;
    for (auto& it : mThreadGroupMap) {
        allThreadGroupKeys.push_back(it.first);
    }
    std::sort(allThreadGroupKeys.begin(), allThreadGroupKeys.end(),
              [](const ThreadGroupKey& a, const ThreadGroupKey& b) { return a.ToString() < b.ToString(); });

    for (auto& key : allThreadGroupKeys) {
        mThreadGroupMap.erase(key);
        mThreadGroupInitStringMap.erase(key);
    }
}

ThreadGroup& ThreadGroupManager::GetThreadGroup(DeviceKind deviceKind, const std::vector<int64_t>& allDeviceId,
                                                int64_t currentDeviceId) {
    int worldSize = static_cast<int>(allDeviceId.size());
    int rank = static_cast<int>(DistributedSpec::GetRankId(currentDeviceId, allDeviceId));
    DAlwaysAssert(rank >= 0);
    DAlwaysAssert(worldSize > 0);

    ThreadGroupKey key;
    ThreadGroupKey::InitThreadGroupKey(key, deviceKind, allDeviceId);
    bool sameDevice = mDTensorInSameDevice || Device::DeviceCount(deviceKind) <= 1;

    // Please note that this function can be executed simultaneously by multiple threads with the same operation op.
    {
        std::unique_lock<std::mutex> lock(mMutex);
        if (mThreadGroupMap.count(key) == 0) {
            std::string initString = "GraphId_" + std::to_string(mGraphId) + "_ThreadGroupHashKey_" +
                                     std::to_string(ParamsHash<ThreadGroupKey>()(key));
            initString = external::boost::ManagedSharedMemory::GetShmFileNameWithPrefix(initString);
            mThreadGroupInitStringMap.emplace(key, initString);
            mThreadGroupMap.emplace(key, worldSize);
        }

        auto it = mThreadGroupMap.find(key);
        DDebugAssert(it->second.size() == static_cast<size_t>(worldSize));
        DDebugAssert(mThreadGroupMap.size() == mThreadGroupInitStringMap.size());
    }

    auto& allRankThreadGroup = mThreadGroupMap.at(key);
    if (allRankThreadGroup[rank] == nullptr) {
        DDebugAssert(mThreadGroupInitStringMap.count(key) > 0);
        auto threadGroup = std::make_unique<ThreadGroup>(mThreadGroupInitStringMap.at(key), key.deviceKind, allDeviceId,
                                                         rank, worldSize, sameDevice);

        threadGroup->Barrier();
        allRankThreadGroup[rank] = std::move(threadGroup);
    }
    DDebugAssert(allRankThreadGroup[rank] != nullptr);

    return *(allRankThreadGroup[rank]);
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
