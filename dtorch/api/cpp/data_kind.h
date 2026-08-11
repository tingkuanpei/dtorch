/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dtorch {
namespace api {
namespace cpp {

enum class DataKind {
    kBool = 0,
    kUInt8,
    kInt8,
    kUInt16,
    kInt16,
    kUInt32,
    kInt32,
    kUInt64,
    kInt64,
    kFloat16,
    kBFloat16,
    kFloat32,
    kFloat64,
    kCount,
};

std::string DataKindToString(DataKind dataKind);

DataKind DataKindFromString(const std::string& str);

std::ostream& operator<<(std::ostream& os, DataKind dataKind);

size_t DataKindSize(DataKind dataKind);

bool DataKindIsInteger(DataKind dataKind);

DataKind DataKindPromote(std::vector<DataKind> inputDataKinds);

// clang-format off
#define DTORCH_FOREACH_DATA_KIND(Func)     \
    Func(api::cpp::DataKind::kFloat16)      \
    Func(api::cpp::DataKind::kFloat32)      \
    Func(api::cpp::DataKind::kFloat64)      \
    Func(api::cpp::DataKind::kBFloat16)     \
    Func(api::cpp::DataKind::kInt8)         \
    Func(api::cpp::DataKind::kInt16)        \
    Func(api::cpp::DataKind::kInt32)        \
    Func(api::cpp::DataKind::kInt64)        \
    Func(api::cpp::DataKind::kUInt8)        \
    Func(api::cpp::DataKind::kUInt16)       \
    Func(api::cpp::DataKind::kUInt32)       \
    Func(api::cpp::DataKind::kUInt64)       \
    Func(api::cpp::DataKind::kBool)
// clang-format on

template <typename DataType>
struct DataKindTrait {};

template <>
struct DataKindTrait<float> {
    static constexpr DataKind value = DataKind::kFloat32;
};

template <>
struct DataKindTrait<double> {
    static constexpr DataKind value = DataKind::kFloat64;
};

template <>
struct DataKindTrait<int8_t> {
    static constexpr DataKind value = DataKind::kInt8;
};

template <>
struct DataKindTrait<int16_t> {
    static constexpr DataKind value = DataKind::kInt16;
};

template <>
struct DataKindTrait<int32_t> {
    static constexpr DataKind value = DataKind::kInt32;
};

template <>
struct DataKindTrait<int64_t> {
    static constexpr DataKind value = DataKind::kInt64;
};

template <>
struct DataKindTrait<uint8_t> {
    static constexpr DataKind value = DataKind::kUInt8;
};

template <>
struct DataKindTrait<uint16_t> {
    static constexpr DataKind value = DataKind::kUInt16;
};

template <>
struct DataKindTrait<uint32_t> {
    static constexpr DataKind value = DataKind::kUInt32;
};

template <>
struct DataKindTrait<uint64_t> {
    static constexpr DataKind value = DataKind::kUInt64;
};

template <>
struct DataKindTrait<bool> {
    static constexpr DataKind value = DataKind::kBool;
};

template <DataKind dataKind>
struct DataTypeTrait {};

template <>
struct DataTypeTrait<DataKind::kFloat16> {
    // TODO: add float16 support
    using DataType = float;
};

template <>
struct DataTypeTrait<DataKind::kFloat32> {
    using DataType = float;
};

template <>
struct DataTypeTrait<DataKind::kFloat64> {
    using DataType = double;
};

template <>
struct DataTypeTrait<DataKind::kBFloat16> {
    // TODO: add bfloat16 support
    using DataType = float;
};

template <>
struct DataTypeTrait<DataKind::kInt8> {
    using DataType = int8_t;
};

template <>
struct DataTypeTrait<DataKind::kInt16> {
    using DataType = int16_t;
};

template <>
struct DataTypeTrait<DataKind::kInt32> {
    using DataType = int32_t;
};

template <>
struct DataTypeTrait<DataKind::kInt64> {
    using DataType = int64_t;
};

template <>
struct DataTypeTrait<DataKind::kUInt8> {
    using DataType = uint8_t;
};

template <>
struct DataTypeTrait<DataKind::kUInt16> {
    using DataType = uint16_t;
};

template <>
struct DataTypeTrait<DataKind::kUInt32> {
    using DataType = uint32_t;
};

template <>
struct DataTypeTrait<DataKind::kUInt64> {
    using DataType = uint64_t;
};

template <>
struct DataTypeTrait<DataKind::kBool> {
    using DataType = bool;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
