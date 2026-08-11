/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <atomic>
#include <cstdint>

#include "dtorch/common/utilities.h"

namespace dtorch {
namespace core {

template <int value>
class GlobalIdManager {
public:
    DTORCH_FORCEINLINE static GlobalIdManager& GetSingleton() {
        static GlobalIdManager singleton;
        return singleton;
    }

    constexpr static uint64_t kNoValue = 0;

public:
    DTORCH_FORCEINLINE uint64_t GetUniqueId() noexcept {
        uint64_t id = mIdAccumulator.fetch_add(1, std::memory_order_relaxed);
        if (id == kNoValue) {
            id = mIdAccumulator.fetch_add(1, std::memory_order_relaxed);
        }
        return id;
    }

private:
    GlobalIdManager() : mIdAccumulator(kNoValue + 1) {}

    ~GlobalIdManager() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(GlobalIdManager);

private:
    std::atomic<uint64_t> mIdAccumulator;
};

// An operator may generate multiple kernels, which will reside on multiple threads, processes, or machines.
// Multiple kernels belonging to the same operator establish communication via the operator ID.
using OperatorIdManager = GlobalIdManager<0>;

// A cluster may create multi graph, each graph has a unique graph ID.
using GraphIdManager = GlobalIdManager<1>;

// TensorPromise shared memory files each need a unique filename.
using TensorPromiseIdManager = GlobalIdManager<2>;

}  // namespace core
}  // namespace dtorch
