/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <atomic>
#include <chrono>

namespace dtorch {

class RandomSeedGenerator {
public:
    static RandomSeedGenerator& Default() {
        static RandomSeedGenerator generator(std::chrono::system_clock::now().time_since_epoch().count());
        return generator;
    }

public:
    explicit RandomSeedGenerator(int64_t seed) : mSeed(seed) {}

    void SetSeed(int64_t seed) { mSeed.store(seed); }

    int64_t NextSeed(int64_t count = 1) { return mSeed.fetch_add(count); }

private:
    std::atomic<int64_t> mSeed;
};

}  // namespace dtorch
