/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "operator_param.h"

#include <array>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

const std::string& OpTypeToString(OperatorType opType) {
    static const std::string kStringVec[] = {
#define DTORCH_FUNC(Name, Value) DTORCH_STRINGIFY(Name),
        DTORCH_FOREACH_OPERATOR_TYPE(DTORCH_FUNC)
#undef DTORCH_FUNC
    };

    static_assert(sizeof(kStringVec) / sizeof(kStringVec[0]) == EnumAsInteger(OperatorType::kCount),
                  "Operator size not equal");
    DDebugAssert(opType < OperatorType::kCount);
    return kStringVec[EnumAsInteger(opType)];
}

OperatorType OpTypeFromString(const std::string& str) {
    static const std::unordered_map<std::string, OperatorType> kStringMap = {
#define DTORCH_FUNC(Name, Value) {DTORCH_STRINGIFY(Name), OperatorType::k##Name},
        DTORCH_FOREACH_OPERATOR_TYPE(DTORCH_FUNC)
#undef DTORCH_FUNC
    };

    DDebugAssert(static_cast<int64_t>(kStringMap.size()) == EnumAsInteger(OperatorType::kCount));

    OperatorType result = OperatorType::kActivation;
    if (kStringMap.count(str) > 0) {
        result = kStringMap.at(str);
    } else {
        throw std::invalid_argument("Unsupported device str: " + str);
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, OperatorType opType) {
    os << OpTypeToString(opType);
    return os;
}

std::vector<int64_t> OpParamUtil::IntOrIntArrayTo2DParam(const IntOrIntArray& intArray,
                                                         const std::string& transferType) {
    std::vector<int64_t> result;

    if (intArray.size() == 1) {
        result = {static_cast<int>(intArray[0]), static_cast<int>(intArray[0])};
    } else if (intArray.size() == 2) {
        result = {static_cast<int>(intArray[0]), static_cast<int>(intArray[1])};
    } else {
        std::stringstream ss;
        ss << "Construct OpParam error, unsupported " << transferType << ", value: " << intArray.ToString();
        throw std::invalid_argument(ss.str());
    }

    return result;
}

std::vector<int64_t> OpParamUtil::IntOrIntArrayTo2DPad(const IntOrIntArray& intArray) {
    std::vector<int64_t> result;

    if (intArray.size() == 1) {
        result = {static_cast<int>(intArray[0]), static_cast<int>(intArray[0]), static_cast<int>(intArray[0]),
                  static_cast<int>(intArray[0])};
    } else if (intArray.size() == 2) {
        result = {static_cast<int>(intArray[0]), static_cast<int>(intArray[1]), static_cast<int>(intArray[0]),
                  static_cast<int>(intArray[1])};
    } else if (intArray.size() == 4) {
        result = intArray.Vec();
    } else {
        throw std::invalid_argument("Construct OpParam error, unsupported pad");
    }

    return result;
}

bool OpParamUtil::IsPadEmptyOrZero(const std::vector<int64_t>& pads) {
    if (pads.empty()) return true;

    for (auto& it : pads) {
        if (it != 0) return false;
    }

    return true;
}

}  // namespace core
}  // namespace dtorch
