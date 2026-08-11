/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "expand_op.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void ExpandOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const auto& param = GetOpParam<ExpandParam>();

    const Shape& inputShape = OperandX()->GetShape();
    Shape& otherShape = const_cast<Shape&>(param.shape);
    for (size_t i = 0; i < otherShape.NumAxis(); i++) {
        size_t otherIdx = otherShape.NumAxis() - i - 1;
        if (otherShape[otherIdx] == -1) {
            DAlwaysAssert(i < inputShape.NumAxis());
            size_t inputIdx = inputShape.NumAxis() - i - 1;
            otherShape[otherIdx] = inputShape[inputIdx];
        }
    }

    if (!inputShape.CanBroadcastWith(otherShape)) {
        std::stringstream ss;
        ss << "Expand op InferOutputShape error, input shape not compatible: ";
        ss << inputShape << " vs " << otherShape;
        throw std::invalid_argument(ss.str());
    }

    Shape outputShape = Shape::BroadcastOutputShape(inputShape, otherShape);

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
}

PlacementSignature ExpandOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    const Shape& inputShape = OperandX()->GetShape();
    const Shape& outputShape = OperandY()->GetShape();
    DDebugAssert(outputShape.NumAxis() >= inputShape.NumAxis());
    size_t expandDimSize = outputShape.NumAxis() - inputShape.NumAxis();

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddOutput("P").Build();

    for (size_t i = 0; i < inputShape.NumAxis(); i++) {
        builder.AddInput(Shard(i)).AddOutput(Shard(expandDimSize + i)).Build();
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
