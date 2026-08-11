/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "operator_serialization_pack.h"

#include <sstream>

#include "dtorch/common/string.h"

namespace dtorch {
namespace core {

std::string OperatorSerializationPack::ToString() const {
    std::stringstream ss;
    ss << "OperatorSerializationPack(opName: " << opName << ", uniqueId: " << uniqueId
       << ", opType: " << opParam->GetOpType() << ", uintInputOperands: " << String::ToString(uintInputOperands)
       << ", uintOutputOperands: " << String::ToString(uintOutputOperands) << ")";
    return ss.str();
}

}  // namespace core
}  // namespace dtorch
