/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "permute_op.h"

#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace dtorch {
namespace core {

void PermuteOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inputShape = OperandX()->GetShape();
    Shape outputShape = inputShape;
    const auto& param = GetOpParam<PermuteParam>();
    if (param.dims.size() != inputShape.NumAxis()) {
        std::stringstream ss;
        ss << "number of dimensions in the tensor input does not match the length of the desired ordering of dimensions"
           << "i.e. input.dim() = " << inputShape.NumAxis() << " is not equal to len(dims) = " << param.dims.size();
        throw std::invalid_argument(ss.str());
    }

    std::unordered_set<size_t> dimSet;
    for (size_t i = 0; i < param.dims.size(); i++) {
        size_t validDim = Operator::GetValidDim(inputShape, param.dims[i]);
        if (dimSet.count(validDim) == 0) {
            dimSet.insert(validDim);
        } else {
            throw std::invalid_argument("duplicate dims are not allowed.");
        }
        outputShape[i] = inputShape[validDim];
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);
}

PlacementSignature PermuteOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);

    const Shape& inputShape = OperandX()->GetShape();
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddOutput("P").Build();

    const auto& param = GetOpParam<PermuteParam>();
    DDebugAssert(param.dims.size() == inputShape.NumAxis());

    for (size_t i = 0; i < param.dims.size(); i++) {
        size_t validDim = Operator::GetValidDim(inputShape, param.dims[i]);
        DDebugAssert(validDim < inputShape.NumAxis());
        builder.AddInput(Shard(validDim)).AddOutput(Shard(i)).Build();
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
