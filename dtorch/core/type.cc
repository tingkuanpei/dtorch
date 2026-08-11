/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "type.h"

#include <array>
#include <vector>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace core {

std::string KernelStreamTypeToString(KernelStreamType streamType) {
    static const std::array<std::string, 2> kStringMap = {
        "compute",
        "communicate",
    };
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(KernelStreamType::kCount),
                  "KernelStreamType size not equal");

    if (streamType == KernelStreamType::kCount) {
        DLogError() << "KernelStreamType invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(streamType)];
}

std::ostream& operator<<(std::ostream& os, KernelStreamType streamType) {
    os << KernelStreamTypeToString(streamType);
    return os;
}

}  // namespace core
}  // namespace dtorch
