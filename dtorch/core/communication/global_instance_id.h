/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <mutex>
#include <string>

#include <unistd.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"

namespace dtorch {
namespace core {
namespace communication {

// In DTorch, communication in FileTensorStore is done via storeName. When we start two DTorch programs on the same
// machine, their storeName will conflict. Therefore, we need to assign a GlobalCommInstanceId to each DTorch instance
// for communication in FileTensorStore. DTorch supports multi-node and multi-process mode, and the GlobalCommInstanceId
// of the same DTorch instance is identical across different machines and processes.

class GlobalCommInstanceId {
public:
    DTORCH_FORCEINLINE static GlobalCommInstanceId& GetSingleton() {
        static GlobalCommInstanceId instance;
        return instance;
    }

public:
    DTORCH_FORCEINLINE const std::string& GetInstanceId() const noexcept {
        std::lock_guard<std::mutex> lock(mMutex);
        DAlwaysAssert(!mInstanceId.empty());
        return mInstanceId;
    }

    DTORCH_FORCEINLINE void SetInstanceId(const std::string& instanceId) noexcept {
        std::lock_guard<std::mutex> lock(mMutex);
        DAlwaysAssert(!instanceId.empty());
        mInstanceId = instanceId;
    }

private:
    GlobalCommInstanceId() : mMutex(), mInstanceId() { mInstanceId = std::to_string(getpid()); }

    ~GlobalCommInstanceId() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(GlobalCommInstanceId);

private:
    mutable std::mutex mMutex;
    std::string mInstanceId;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
