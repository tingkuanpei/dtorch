/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "repeat_interleave_op.h"

namespace dtorch {
namespace core {

void RepeatInterleaveOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const auto& param = GetOpParam<RepeatInterleaveParam>();
    const Shape& inShape = OperandX()->GetShape();
    Shape outShape = inShape;

    if (param.repeats < 0) {
        throw std::invalid_argument("Repeats must be non-negative");
    }

    if (!param.dim.has_value()) {
        outShape = Shape();
        outShape.PushBack(inShape.Count() * param.repeats);
    } else {
        size_t dim = Operator::GetValidDim(outShape, param.dim.value());
        outShape[dim] = outShape[dim] * param.repeats;
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outShape);
}

PlacementSignature RepeatInterleaveOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);
    const auto& param = GetOpParam<RepeatInterleaveParam>();
    const Shape& inShape = OperandX()->GetShape();
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());

    builder.AddInput("P").AddOutput("P").Build();
    if (param.dim.has_value()) {
        size_t dim = Operator::GetValidDim(inShape, param.dim.value());
        for (size_t i = 0; i < inShape.NumAxis(); i++) {
            if (i != dim) {
                builder.AddInput(Shard(i)).AddOutput(Shard(i)).Build();
            }
        }
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
