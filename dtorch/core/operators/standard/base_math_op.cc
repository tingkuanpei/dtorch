/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "base_math_op.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void BaseMathOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() >= 1 && GetOutputSize() == 1);
    const auto& param = GetOpParam<BaseMathParam>();
    IgnoreUnused(param);

    OperandY()->MetaDataSameAs(OperandX());

    std::unordered_set<BaseMathType> boolOutputKind = {BaseMathType::kIsInf, BaseMathType::kIsNan};
    if (boolOutputKind.find(param.baseMathType) != boolOutputKind.end()) {
        OperandY()->SetDataKind(DataKind::kBool);
    }
}

}  // namespace core
}  // namespace dtorch
