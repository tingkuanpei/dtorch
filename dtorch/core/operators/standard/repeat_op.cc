/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "repeat_op.h"

#include <stdexcept>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void RepeatOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inShape = OperandX()->GetShape();

    const auto& param = GetOpParam<RepeatParam>();
    if (param.repeat.size() < inShape.NumAxis()) {
        throw std::invalid_argument(
            "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
    }

    DDebugAssert(param.repeat.size() >= inShape.NumAxis());
    Shape inShapeExpand = inShape.ExpandBefore(param.repeat.size());
    Shape outShape;
    for (size_t i = 0; i < inShapeExpand.NumAxis(); i++) {
        if (param.repeat[i] < 0) {
            throw std::invalid_argument("Repeat op's repeat parameter less than zero");
        }
        outShape.PushBack(inShapeExpand[i] * param.repeat[i]);
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outShape);
}

}  // namespace core
}  // namespace dtorch
