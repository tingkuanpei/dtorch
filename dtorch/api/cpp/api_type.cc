/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "api_type.h"

#include <array>
#include <unordered_map>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace api {
namespace cpp {

std::string OperatorFormatToString(OperatorFormat operatorFormat) {
    static const std::array<std::string, 2> kStringMap = {
        "nchw",
        "nhwc",
    };
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(OperatorFormat::kCount),
                  "OperatorFormat size not equal");

    if (operatorFormat == OperatorFormat::kCount) {
        DLogError() << "operatorFormat invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(operatorFormat)];
}

std::ostream& operator<<(std::ostream& os, OperatorFormat operatorFormat) {
    os << OperatorFormatToString(operatorFormat);
    return os;
}

bool StringToOperatorFormat(const std::string& str, OperatorFormat& operatorFormat) {
    std::unordered_map<std::string, api::cpp::OperatorFormat> operatorFormatMap = {
        {"kNCHW", api::cpp::OperatorFormat::kNCHW},
        {"kNHWC", api::cpp::OperatorFormat::kNHWC},
    };
    auto it = operatorFormatMap.find(str);
    if (it == operatorFormatMap.end()) {
        return false;
    }

    operatorFormat = it->second;
    return true;
}

std::string PaddingTypeToString(PaddingType paddingType) {
    static const std::array<std::string, 3> kStringMap = {
        "not_set",
        "same",
        "valid",
    };
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(PaddingType::kCount),
                  "PaddingType size not equal");

    if (paddingType == PaddingType::kCount) {
        DLogError() << "paddingType invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(paddingType)];
}

std::ostream& operator<<(std::ostream& os, PaddingType paddingType) {
    os << PaddingTypeToString(paddingType);
    return os;
}

bool StringToPaddingType(const std::string& str, PaddingType& paddingType) {
    std::unordered_map<std::string, api::cpp::PaddingType> paddingTypeMap = {
        {"kNotSet", api::cpp::PaddingType::kNotSet},
        {"kSame", api::cpp::PaddingType::kSame},
        {"kValid", api::cpp::PaddingType::kValid},
    };
    auto it = paddingTypeMap.find(str);
    if (it == paddingTypeMap.end()) {
        return false;
    }

    paddingType = it->second;
    return true;
}

std::string PoolingKindToString(PoolingKind poolingKind) {
    static const std::array<std::string, 2> kStringMap = {
        "avg",
        "max",
    };
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(PoolingKind::kCount),
                  "poolingKind size not equal");

    if (poolingKind == PoolingKind::kCount) {
        DLogError() << "poolingKind invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(poolingKind)];
}

std::ostream& operator<<(std::ostream& os, PoolingKind PoolingKind) {
    os << PoolingKindToString(PoolingKind);
    return os;
}

bool StringToPoolingKind(const std::string& str, PoolingKind& poolingKind) {
    std::unordered_map<std::string, api::cpp::PoolingKind> poolingKindMap = {
        {"kAvg", api::cpp::PoolingKind::kAvg},
        {"kMax", api::cpp::PoolingKind::kMax},
    };
    auto it = poolingKindMap.find(str);
    if (it == poolingKindMap.end()) {
        return false;
    }

    poolingKind = it->second;
    return true;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
