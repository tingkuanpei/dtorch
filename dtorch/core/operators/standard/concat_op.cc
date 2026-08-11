/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "concat_op.h"

namespace dtorch {
namespace core {

void ConcatOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() > 0 && GetOutputSize() == 1);
    size_t inputSize = GetInputSize();
    const auto& param = GetOpParam<ConcatParam>();
    const Shape& inputShape = GetInputOperand(0)->GetShape();
    size_t dim = Operator::GetValidDim(inputShape, param.dim);
    size_t outputSize = inputShape[dim];

    for (size_t i = 1; i < inputSize; i++) {
        Shape tmpShape = GetInputOperand(i)->GetShape();
        outputSize += tmpShape[dim];
        tmpShape[dim] = inputShape[dim];
        if (tmpShape != inputShape) {
            std::stringstream ss;
            ss << "Concat op InferOutputShape error, input shape must be compatible, but get: " << inputShape << " vs "
               << GetInputOperand(i)->GetShape();
            throw std::invalid_argument(ss.str());
        }
    }

    Shape outputShape = inputShape;
    outputShape[dim] = outputSize;
    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
}

PlacementSignature ConcatOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() > 0 && GetOutputSize() == 1);
    size_t inputSize = GetInputSize();
    const Shape& inputShape = GetInputOperand(0)->GetShape();
    const auto& param = GetOpParam<ConcatParam>();
    size_t dim = Operator::GetValidDim(inputShape, param.dim);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    for (size_t n = 0; n < inputSize; n++) {
        builder.AddInput("P");
    }
    builder.AddOutput("P").Build();

    for (size_t i = 0; i < inputShape.NumAxis(); i++) {
        if (i == dim) {
            continue;
        }

        for (size_t n = 0; n < inputSize; n++) {
            builder.AddInput(Shard(i));
        }
        builder.AddOutput(Shard(i)).Build();
    }
    if (inputSize == 2) {
        builder.AddInput(Shard(dim)).AddInput(Shard(dim)).AddOutput(Shard(dim, inputShape[dim] - 1)).Build();
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
