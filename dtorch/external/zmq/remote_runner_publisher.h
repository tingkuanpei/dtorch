/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "dtorch/core/runner/node_runner_base.h"

namespace dtorch {
namespace external {
namespace zmq {

class RemoteRunnerPublisher {
public:
    inline static const std::string kExecuteStr = "publisherExecute";
    inline static const std::string kDestroyStr = "publisherDestroy";

public:
    RemoteRunnerPublisher(const std::string& address);

    ~RemoteRunnerPublisher();

    void Execute(const std::vector<std::shared_ptr<core::Operator>>& ops,
                 const std::vector<const core::Operand*>& noHoldOperands);

    void SendDestroy();

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

class PublishMessageIdManager {
public:
    DTORCH_FORCEINLINE static PublishMessageIdManager& GetSingleton() {
        static PublishMessageIdManager singleton;
        return singleton;
    }

    constexpr static int64_t kInitValue = -1;

public:
    DTORCH_FORCEINLINE int64_t GetIdAndIncrement(const std::string& address) noexcept {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mMessageIdMap.find(address) == mMessageIdMap.end()) {
            mMessageIdMap[address] = kInitValue;
        }
        return ++mMessageIdMap[address];
    }

    DTORCH_FORCEINLINE int64_t GetId(const std::string& address) noexcept {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mMessageIdMap.find(address) == mMessageIdMap.end()) {
            mMessageIdMap[address] = kInitValue;
        }
        return mMessageIdMap[address];
    }

private:
    PublishMessageIdManager() : mMutex(), mMessageIdMap() {}

    ~PublishMessageIdManager() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(PublishMessageIdManager);

private:
    std::mutex mMutex;
    std::unordered_map<std::string, int64_t> mMessageIdMap;
};

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
