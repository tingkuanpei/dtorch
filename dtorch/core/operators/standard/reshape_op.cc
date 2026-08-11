/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "reshape_op.h"

#include <cstddef>
#include <cstdint>
#include <sstream>

namespace dtorch {
namespace core {

bool ReshapeOp::CalculateOutputShape(const Shape& inputShape, const Shape& expectShape, Shape& outputShape) {
    outputShape = expectShape;

    bool minusOneFlags = false;
    int minusOneIndex = 0;
    size_t sum = 1;

    for (int i = 0; i < static_cast<int>(outputShape.NumAxis()); i++) {
        if (outputShape.At(i) == 0) {
            if (i >= static_cast<int>(inputShape.NumAxis())) {
                return false;
            }

            outputShape[i] = inputShape[i];
            sum *= outputShape[i];
        } else if (outputShape.At(i) == -1) {
            if (minusOneFlags) {
                return false;
            }
            minusOneFlags = true;
            minusOneIndex = i;
        } else if (outputShape.At(i) <= -2) {
            return false;
        } else {
            sum *= outputShape[i];
        }
    }

    if (minusOneFlags) {
        outputShape[minusOneIndex] = inputShape.Count() / sum;
        if (inputShape.Count() % sum != 0) {
            return false;
        }
    }

    return true;
}

void ReshapeOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1 || GetInputSize() == 2);
    const auto& param = GetOpParam<ReshapeParam>();

    Shape inputShape = OperandX()->GetShape();
    Shape expectShape = param.shape;
    Shape outputShape;

    if (!CalculateOutputShape(inputShape, expectShape, outputShape)) {
        std::stringstream ss;
        ss << "Reshape op InferOutputShape error, inputShape: " << inputShape << "  expectShape: " << expectShape;

        throw std::invalid_argument(ss.str());
    }

    if (outputShape.Count() != inputShape.Count()) {
        std::stringstream ss;
        ss << "Reshape op InferOutputShape error, inputShape: " << inputShape << "  expectShape: " << expectShape
           << " calculate outputShape: " << outputShape;

        throw std::invalid_argument(ss.str());
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
}

std::string ReshapeOp::GetDescribeString() const {
    const auto& param = GetOpParam<ReshapeParam>();
    std::stringstream ss;
    ss << GetOpType() << ": expectShape: " << param.shape;
    return ss.str();
}

PlacementSignature ReshapeOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    const Shape& inputShape = OperandX()->GetShape();
    const Shape& outputShape = OperandY()->GetShape();

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddOutput("P").Build();

    for (size_t i = 0; i < inputShape.NumAxis() && i < outputShape.NumAxis(); i++) {
        builder.AddInput(Shard(i)).AddOutput(Shard(i)).Build();
        if (inputShape[i] != outputShape[i]) {
            break;
        }
    }

    for (size_t i = 0; i < inputShape.NumAxis() && i < outputShape.NumAxis(); i++) {
        int64_t inIndex = inputShape.NumAxis() - 1 - i;
        int64_t outIndex = outputShape.NumAxis() - 1 - i;
        if (inputShape[inIndex] == outputShape[outIndex]) {
            builder.AddInput(Shard(inIndex)).AddOutput(Shard(outIndex)).Build();
        } else {
            break;
        }
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
