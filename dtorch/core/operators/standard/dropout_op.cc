/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "dropout_op.h"

namespace dtorch {
namespace core {

void DropoutOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    DDebugAssert(GetOutputSize() == 1 || GetOutputSize() == 2);
    const auto& param = GetOpParam<DropoutParam>();
    if (param.probability < 0 || param.probability >= 1.0f) {
        throw std::invalid_argument("probability must in [0, 1), current value is: " +
                                    std::to_string(param.probability));
    }

    OperandY()->MetaDataSameAs(OperandX());
}

}  // namespace core
}  // namespace dtorch
