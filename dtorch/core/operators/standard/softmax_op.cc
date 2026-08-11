/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "softmax_op.h"

namespace dtorch {
namespace core {

void SoftmaxOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);

    const auto& param = GetOpParam<SoftmaxParam>();
    // check dim valid or not
    Operator::GetValidDim(OperandX()->GetShape(), param.dim);

    OperandY()->MetaDataSameAs(OperandX());
}

PlacementSignature SoftmaxOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    const auto& param = GetOpParam<SoftmaxParam>();
    const Shape& inputShape = OperandX()->GetShape();
    size_t dim = Operator::GetValidDim(inputShape, param.dim);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    for (size_t i = 0; i <= inputShape.NumAxis(); i++) {
        if (dim == i) {
            continue;
        }
        builder.AddInput(Shard(i)).AddOutput(Shard(i)).Build();
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
