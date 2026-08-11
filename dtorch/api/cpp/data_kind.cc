/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "data_kind.h"

#include <array>
#include <map>
#include <sstream>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace api {
namespace cpp {

std::string DataKindToString(DataKind dataKind) {
    static const std::array<std::string, 13> kStringMap = {
        "bool",   "uint8", "int8",    "uint16",   "int16",   "uint32",  "int32",
        "uint64", "int64", "float16", "bfloat16", "float32", "float64",
    };
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(DataKind::kCount), "Data kind size not equal");

    if (dataKind == DataKind::kCount) {
        DLogFatal() << "data Kind invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(dataKind)];
}

DataKind DataKindFromString(const std::string& str) {
    static const std::map<std::string, DataKind> kStringMap = {
        {"bool", DataKind::kBool},       {"uint8", DataKind::kUInt8},       {"int8", DataKind::kInt8},
        {"uint16", DataKind::kUInt16},   {"int16", DataKind::kInt16},       {"uint32", DataKind::kUInt32},
        {"int32", DataKind::kInt32},     {"uint64", DataKind::kUInt64},     {"int64", DataKind::kInt64},
        {"float16", DataKind::kFloat16}, {"bfloat16", DataKind::kBFloat16}, {"float32", DataKind::kFloat32},
        {"float64", DataKind::kFloat64},
    };
    DDebugAssertMsg(static_cast<int>(kStringMap.size()) == EnumAsInteger(DataKind::kCount), "Data kind size not equal");

    auto it = kStringMap.find(str);
    if (it == kStringMap.end()) {
        DLogFatal() << "data kind " << str << " not found";
        return DataKind::kCount;
    }

    return it->second;
}

std::ostream& operator<<(std::ostream& os, DataKind dataKind) {
    os << DataKindToString(dataKind);
    return os;
}

size_t DataKindSize(DataKind dataKind) {
    static constexpr std::array<size_t, 13> kSizeMap = {1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 2, 4, 8};
    static_assert(static_cast<int>(kSizeMap.size()) == EnumAsInteger(DataKind::kCount), "Data kind size not equal");

    DDebugAssert(dataKind < DataKind::kCount);
    return kSizeMap[EnumAsInteger(dataKind)];
}

bool DataKindIsInteger(DataKind dataKind) {
    if (dataKind == DataKind::kFloat16 || dataKind == DataKind::kFloat32 || dataKind == DataKind::kFloat64 ||
        dataKind == DataKind::kBFloat16) {
        return false;
    }

    return true;
}

DataKind DataKindPromote(std::vector<DataKind> inputDataKinds) {
    // TODO: fix with zero-dim operands
    // TODO: fix with mul
    // TODO: Promotion for uint16, uint32, uint64 types is not supported, attempted to promote Int and UInt32
    // https://docs.pytorch.org/docs/stable/tensor_attributes.html#id4

    DDebugAssert(inputDataKinds.size() > 0);

    int maxValue = 0;
    for (auto dataKind : inputDataKinds) {
        if (EnumAsInteger(dataKind) > maxValue) {
            maxValue = EnumAsInteger(dataKind);
        }
    }

    return static_cast<DataKind>(maxValue);
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
