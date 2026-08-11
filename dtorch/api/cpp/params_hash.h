/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstring>
#include <memory>
#include <type_traits>

// This file is modified from PyTorch, with tag: 1.10.1, commit id: 302ee7bfb60
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/utils/ParamsHash.h

namespace dtorch {
namespace api {
namespace cpp {

// Hashing machinery for Params
// Fowler–Noll–Vo hash function
// see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Params>
struct ParamsHash {
    // Params must be a POD because we read out its memory
    // contenst as char* when hashing
    static_assert(std::is_pod<Params>::value, "Params is not POD");

    size_t operator()(const Params& params) const {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&params);
        uint32_t value = 0x811C9DC5;
        for (int i = 0; i < (int)sizeof(Params); i++) {
            value ^= ptr[i];
            value *= 0x01000193;
        }
        return (size_t)value;
    }
};

template <typename Params>
struct ParamsEqual {
    // Params must be a POD because we read out its memory
    // contenst as char* when comparing
    static_assert(std::is_pod<Params>::value, "Params is not POD");

    bool operator()(const Params& a, const Params& b) const {
        const uint8_t* ptr1 = reinterpret_cast<const uint8_t*>(&a);
        const uint8_t* ptr2 = reinterpret_cast<const uint8_t*>(&b);
        return std::memcmp(ptr1, ptr2, sizeof(Params)) == 0;
    }
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
