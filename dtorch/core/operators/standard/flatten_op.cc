/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "flatten_op.h"

#include <sstream>

namespace dtorch {
namespace core {

bool FlattenOp::CalculateOutputShape(const Shape& inputShape, const FlattenParam& param, Shape& outputShape) {
    DDebugAssert(outputShape.Empty());

    size_t startDim = Operator::GetValidDim(inputShape, param.startDim);
    size_t endDim = Operator::GetValidDim(inputShape, param.endDim);
    if (startDim > endDim) {
        std::stringstream ss;
        ss << "Flatten op invalid start & end dimension, input shape: " << inputShape
           << "  startDim: " << param.startDim << "  endDim: " << param.endDim;
        throw std::invalid_argument(ss.str());
    }

    for (size_t i = 0; i < startDim; i++) {
        outputShape.PushBack(inputShape[i]);
    }

    size_t sum = 1;
    for (size_t i = startDim; i <= endDim; i++) {
        sum *= inputShape[i];
    }
    outputShape.PushBack(sum);

    for (size_t i = endDim + 1; i < inputShape.NumAxis(); i++) {
        outputShape.PushBack(inputShape[i]);
    }

    return true;
}

void FlattenOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const auto& param = GetOpParam<FlattenParam>();

    Shape inputShape = OperandX()->GetShape();
    Shape outputShape;

    if (!CalculateOutputShape(inputShape, param, outputShape)) {
        std::stringstream ss;
        ss << "Flatten op InferOutputShape error, inputShape: " << inputShape << "  startDim: " << param.startDim
           << "  endDim: " << param.endDim;
        throw std::invalid_argument(ss.str());
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
}

std::string FlattenOp::GetDescribeString() const {
    const auto& param = GetOpParam<FlattenParam>();
    std::stringstream ss;
    ss << GetOpType() << ": startDim: " << param.startDim << " endDim: " << param.endDim;
    return ss.str();
}

PlacementSignature FlattenOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    const Shape& inputShape = OperandX()->GetShape();

    const auto& param = GetOpParam<FlattenParam>();
    size_t startDim = Operator::GetValidDim(inputShape, param.startDim);
    size_t endDim = Operator::GetValidDim(inputShape, param.endDim);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddOutput("P").Build();
    for (size_t i = 0; i <= startDim; i++) {
        builder.AddInput(Shard(i)).AddOutput(Shard(i)).Build();
    }

    for (size_t i = endDim + 1; i < inputShape.NumAxis(); i++) {
        builder.AddInput(Shard(i)).AddOutput(Shard(i - (endDim - startDim + 1))).Build();
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
