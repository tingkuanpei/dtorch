/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "transpose_op.h"

namespace dtorch {
namespace core {

void TransposeOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inputShape = OperandX()->GetShape();
    const auto& param = GetOpParam<TransposeParam>();
    size_t dim0 = Operator::GetValidDim(inputShape, param.dim0);
    size_t dim1 = Operator::GetValidDim(inputShape, param.dim1);

    Shape outputShape = inputShape;
    outputShape[dim0] = inputShape[dim1];
    outputShape[dim1] = inputShape[dim0];

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
}

PlacementSignature TransposeOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    const Shape& inputShape = OperandX()->GetShape();
    const auto& param = GetOpParam<TransposeParam>();
    size_t dim0 = Operator::GetValidDim(inputShape, param.dim0);
    size_t dim1 = Operator::GetValidDim(inputShape, param.dim1);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddOutput("P").Build();
    for (size_t i = 0; i < inputShape.NumAxis(); i++) {
        if (i == dim0) {
            builder.AddInput(Shard(dim0)).AddOutput(Shard(dim1)).Build();
        } else if (i == dim1) {
            builder.AddInput(Shard(dim1)).AddOutput(Shard(dim0)).Build();
        } else {
            builder.AddInput(Shard(i)).AddOutput(Shard(i)).Build();
        }
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
