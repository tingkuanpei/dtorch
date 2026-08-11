/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "dtorch/common/debug.h"
#include "dtorch/core/operators/operator.h"
#include "dtorch/core/type.h"
#include "thread_group.h"

namespace dtorch {
namespace core {
namespace communication {

// ThreadGroupKey MUST POD type
struct ThreadGroupKey {
    DeviceKind deviceKind;
    int64_t allDeviceId[12];

    static void InitThreadGroupKey(ThreadGroupKey& threadGroupKey, DeviceKind deviceKind,
                                   const std::vector<int64_t>& allDeviceId) {
        // KernelStreamKey maybe align, set it all to 0
        std::memset(&threadGroupKey, 0, sizeof(ThreadGroupKey));

        threadGroupKey.deviceKind = deviceKind;
        DAlwaysAssert(allDeviceId.size() <= 12);
        std::memcpy(threadGroupKey.allDeviceId, allDeviceId.data(), allDeviceId.size() * sizeof(int64_t));
    }

    std::string ToString() const {
        std::stringstream ss;
        ss << "ThreadGroupKey(" << DeviceKindToString(deviceKind) << ", allDeviceId:[";
        for (int64_t deviceId : allDeviceId) {
            ss << deviceId << ",";
        }
        ss << "]";
        return ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const ThreadGroupKey& threadGroupKey) {
        os << threadGroupKey.ToString();
        return os;
    }
};

// eg: SimpleArray = [[0, 1, 2, 3], [4, 5, 6, 7]] dim = 0, will get 2 group
//     group 0 = [0, 1, 2, 3], group 1 = [4, 5, 6, 7]
// if deviceId == 5, will return mSubGroup = 1, mAllDeviceId = [[4, 5, 6, 7]

struct ThreadGroupInfo {
    ThreadGroupInfo(const SimpleArray& array, size_t dim, int64_t deviceId);

    DTORCH_API_FORCEINLINE size_t GetSubGroupId() const noexcept { return mSubGroup; }

    DTORCH_API_FORCEINLINE std::vector<int64_t> GetAllDeviceIds() const noexcept { return mAllDeviceId; }

public:
    size_t mSubGroup;
    std::vector<int64_t> mAllDeviceId;
};

class ThreadGroupManager {
public:
    ThreadGroupManager(uint64_t graphId, bool dtensorInSameDevice)
        : mMutex(),
          mGraphId(graphId),
          mDTensorInSameDevice(dtensorInSameDevice),
          mThreadGroupMap(),
          mThreadGroupInitStringMap() {}

    ~ThreadGroupManager();

    DTORCH_DISABLE_COPY_AND_MOVE(ThreadGroupManager);

    // Threads with the same op can run concurrently, while threads with other op will wait.
    // One operator call get multi thread-group, use subGroupId to identity it.
    ThreadGroup& GetThreadGroup(DeviceKind deviceKind, const std::vector<int64_t>& allDeviceId,
                                int64_t currentDeviceId);

private:
    std::mutex mMutex;
    uint64_t mGraphId;
    bool mDTensorInSameDevice;
    std::unordered_map<ThreadGroupKey, std::vector<std::unique_ptr<ThreadGroup>>, ParamsHash<ThreadGroupKey>,
                       ParamsEqual<ThreadGroupKey>>
        mThreadGroupMap;
    std::unordered_map<ThreadGroupKey, std::string, ParamsHash<ThreadGroupKey>, ParamsEqual<ThreadGroupKey>>
        mThreadGroupInitStringMap;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
