/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "unsqueeze_op.h"

namespace dtorch {
namespace core {

void UnsqueezeOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inShape = OperandX()->GetShape();

    const auto& param = GetOpParam<UnsqueezeParam>();

    Shape outShape(inShape.NumAxis() + 1, 1);
    size_t dim = Operator::GetValidDim(outShape, param.dim);
    for (size_t i = 0, readIndex = 0; i < outShape.NumAxis(); i++) {
        if (i != dim) {
            outShape[i] = inShape[readIndex];
            readIndex++;
        } else {
            outShape[i] = 1;
        }
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outShape);
}

PlacementSignature UnsqueezeOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddOutput("P").Build();

    const auto& param = GetOpParam<UnsqueezeParam>();
    const Shape& inShape = OperandX()->GetShape();
    size_t dim = Operator::GetValidDim(inShape, param.dim);
    for (size_t i = 0; i < inShape.NumAxis(); i++) {
        size_t outIdx = i < dim ? i : i + 1;
        builder.AddInput(Shard(i)).AddOutput(Shard(outIdx)).Build();
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
