/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "where_op.h"

#include <sstream>
#include <stdexcept>

#include "dtorch/common/debug.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

void WhereOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 3);
    const Shape& conditionShape = OperandA()->GetShape();
    const Shape& inputShape = OperandB()->GetShape();
    const Shape& otherShape = OperandC()->GetShape();
    const auto& param = GetOpParam<WhereParam>();
    Shape outputShape = conditionShape;
    DataKind outputDataKind = DataKind::kFloat32;

    if (OperandA()->GetDataKind() != DataKind::kBool) {
        std::stringstream ss;
        ss << "where expected condition to be a boolean tensor, but got a tensor with dtype "
           << OperandA()->GetDataKind();
        throw std::invalid_argument(ss.str());
    }

    if (!inputShape.IsNullTensorShape()) {
        DDebugAssert(!param.input.has_value());
        if (!outputShape.CanBroadcastWith(inputShape)) {
            std::stringstream ss;
            ss << "WhereOp op InferOutputShape error, shape not compatible: ";
            ss << outputShape.ToString() << " vs " << inputShape.ToString();
            throw std::invalid_argument(ss.str());
        }

        outputShape = Shape::BroadcastOutputShape(outputShape, inputShape);
        outputDataKind = OperandB()->GetDataKind();
    } else {
        DDebugAssert(param.input.has_value());
    }

    if (!otherShape.IsNullTensorShape()) {
        DDebugAssert(!param.other.has_value());
        if (!outputShape.CanBroadcastWith(otherShape)) {
            std::stringstream ss;
            ss << "WhereOp op InferOutputShape error, shape not compatible: ";
            ss << outputShape.ToString() << " vs " << otherShape.ToString();
            throw std::invalid_argument(ss.str());
        }

        outputShape = Shape::BroadcastOutputShape(outputShape, otherShape);
        outputDataKind = OperandC()->GetDataKind();
    } else {
        DDebugAssert(param.other.has_value());
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
    OperandY()->SetDataKind(outputDataKind);
}

PlacementSignature WhereOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 3);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    // TODO:

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
