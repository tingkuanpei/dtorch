/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "clamp_op.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void ClampOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 3);

    const auto& param = GetOpParam<ClampParam>();
    const Shape& inputShape = OperandA()->GetShape();
    const Shape& minTensorShape = OperandB()->GetShape();
    const Shape& maxTensorShape = OperandC()->GetShape();

    if (param.min.has_value() || param.max.has_value()) {
        DAlwaysAssert(minTensorShape.IsNullTensorShape() && maxTensorShape.IsNullTensorShape());
    }
    if ((!minTensorShape.IsNullTensorShape()) || (!maxTensorShape.IsNullTensorShape())) {
        DAlwaysAssert((!param.min.has_value()) && (!param.max.has_value()));
    }

    if (!minTensorShape.IsNullTensorShape()) {
        DDebugAssert(!maxTensorShape.IsNullTensorShape());
        if (!inputShape.CanBroadcastWith(minTensorShape)) {
            std::stringstream ss;
            ss << "Clamp op InferOutputShape error, input tensor shape not compatible with min tensor shape: ";
            ss << inputShape << " vs " << minTensorShape;
            throw std::invalid_argument(ss.str());
        }
        if (!inputShape.CanBroadcastWith(maxTensorShape)) {
            std::stringstream ss;
            ss << "Clamp op InferOutputShape error, input tensor shape not compatible with max tensor shape: ";
            ss << inputShape << " vs " << maxTensorShape;
            throw std::invalid_argument(ss.str());
        }
    }

    OperandY()->MetaDataSameAs(OperandX());
}

}  // namespace core
}  // namespace dtorch
