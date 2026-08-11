/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "max_min_op.h"

#include "dtorch/core/operand.h"

namespace dtorch {
namespace core {

size_t MaxMinOp::InferOutputSize() const {
    const auto& param = GetOpParam<MaxMinParam>();
    if (param.dim.has_value()) {
        return 2;
    } else {
        return 1;
    }
}

void MaxMinOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inShape = OperandX()->GetShape();
    const auto& param = GetOpParam<MaxMinParam>();

    Shape outShape;
    if (param.dim.has_value()) {
        outShape = inShape;
        size_t dim = Operator::GetValidDim(inShape, param.dim.value());
        if (param.keepdim) {
            outShape[dim] = 1;
        } else {
            outShape = outShape.RemoveSelectedDim({dim});
        }
    }

    DDebugAssert(GetOutputSize() == 1 || GetOutputSize() == 2);
    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outShape);

    if (GetOutputSize() == 2) {
        Operand* indices = GetOutputOperand(1);
        indices->MetaDataSameAs(OperandX());
        indices->SetDataKind(DataKind::kInt64);
        indices->SetShapeAndStride(outShape);
    }
}

PlacementSignature MaxMinOp::GetPlacementSignature() const {
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    // TODO:

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
