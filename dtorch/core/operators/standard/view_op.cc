/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "view_op.h"

namespace dtorch {
namespace core {

void ViewOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const auto& param = GetOpParam<ViewParam>();

    if (param.shape.has_value()) {
        const Shape& inputShape = OperandX()->GetShape();
        Shape outputShape = param.shape.value();

        int64_t accumOfElement = 1;
        int64_t minusOneIdx = -1;
        for (size_t i = 0; i < outputShape.NumAxis(); i++) {
            if (outputShape[i] > 0) {
                accumOfElement *= outputShape[i];
            } else if (outputShape[i] == -1) {
                DDebugAssert(minusOneIdx == -1);
                minusOneIdx = i;
            }
        }

        if (minusOneIdx != -1) {
            outputShape[minusOneIdx] = inputShape.Count() / accumOfElement;
        }

        OperandY()->MetaDataSameAs(OperandX());
        OperandY()->SetShapeAndStride(outputShape);
    } else {
        DDebugAssert(param.placements.has_value());
        PlacementSeq placementSeq = param.placements.value();

        const Shape& inputShape = OperandX()->GetShape();
        const DeviceMesh inputDeviceMesh = OperandX()->GetDeviceMesh();
        if (!DistributedSpec::CheckShapeValid(inputShape, inputDeviceMesh, placementSeq)) {
            std::stringstream ss;
            ss << "View operator invalid placement: " << placementSeq;
            throw std::invalid_argument(ss.str());
        }

        OperandY()->MetaDataSameAs(OperandX());
        OperandY()->SetDeviceMeshAndPlacementSeq(inputDeviceMesh, placementSeq);
    }
}

PlacementSignature ViewOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddOutput("P").Build();

    const Shape& inputShape = OperandX()->GetShape();
    const Shape& outputShape = OperandY()->GetShape();
    const auto& param = GetOpParam<ViewParam>();
    DDebugAssert(param.shape.has_value());
    Shape expectOutputShape = param.shape.value();

    for (size_t i = 0; i < inputShape.NumAxis() && i < outputShape.NumAxis(); i++) {
        if (inputShape[i] == outputShape[i]) {
            builder.AddInput(Shard(i)).AddOutput(Shard(i)).Build();
        } else {
            // TODO: need some check
            builder.AddInput(Shard(i)).AddOutput(Shard(i)).Build();
            break;
        }
    }

    for (size_t i = 1; i <= inputShape.NumAxis() && i <= outputShape.NumAxis(); i++) {
        size_t inIdx = inputShape.NumAxis() - i;
        size_t outIdx = outputShape.NumAxis() - i;

        if (inputShape[inIdx] == outputShape[outIdx]) {
            builder.AddInput(Shard(inIdx)).AddOutput(Shard(outIdx)).Build();
        } else {
            break;
        }
    }

    return builder.Finish();
}

std::string ViewOp::GetDescribeString() const {
    DDebugAssert(GetInputSize() >= 1);
    const auto& param = GetOpParam<ViewParam>();

    std::stringstream ss;
    if (param.shape.has_value()) {
        ss << GetOpType() << ": input shape: " << OperandX()->GetShape() << ", param.shape : " << param.shape.value();
    } else {
        ss << GetOpType() << ": input placements: " << OperandX()->GetPlacementSeq()
           << ", param.placements : " << param.placements.value();
    }

    return ss.str();
}

}  // namespace core
}  // namespace dtorch
