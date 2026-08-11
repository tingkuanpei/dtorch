/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "squeeze_op.h"

#include "dtorch/common/string.h"

namespace dtorch {
namespace core {

void SqueezeOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inShape = OperandX()->GetShape();

    const auto& param = GetOpParam<SqueezeParam>();
    std::unordered_set<int> dimSet;
    for (auto it : param.dims) {
        dimSet.insert(Operator::GetValidDim(inShape, it));
    }

    Shape outShape;
    for (size_t i = 0; i < inShape.NumAxis(); i++) {
        if (dimSet.empty() || dimSet.find(static_cast<int>(i)) != dimSet.end()) {
            if (inShape[i] != 1) {
                mIODimMap[i] = outShape.NumAxis();
                outShape.PushBack(inShape[i]);
            }
        } else {
            mIODimMap[i] = outShape.NumAxis();
            outShape.PushBack(inShape[i]);
        }
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outShape);
}

PlacementSignature SqueezeOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddOutput("P").Build();
    for (auto it : mIODimMap) {
        builder.AddInput(Shard(it.first)).AddOutput(Shard(it.second)).Build();
    }

    return builder.Finish();
}

std::string SqueezeOp::GetDescribeString() const {
    DDebugAssert(GetInputSize() >= 1);
    const auto& param = GetOpParam<SqueezeParam>();
    const Shape& inShape = OperandX()->GetShape();

    std::stringstream ss;
    ss << GetOpType() << ": input shape: " << inShape << ", param.dims : " << String::ToString(param.dims);
    return ss.str();
}

}  // namespace core
}  // namespace dtorch
