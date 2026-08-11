/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "reduce_op.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

ReduceParam::ReduceParam(ReduceKind reduceKind, const IntOrIntArray& dim, bool keepdim,
                         std::optional<DataKind> dataKind)
    : OpParam(OperatorType::kReduce), reduceKind(reduceKind), dim(dim.Vec()), keepdim(keepdim), dataKind(dataKind) {}

std::unordered_set<size_t> ReduceParam::GetDimSet(const Shape& inShape) const {
    std::unordered_set<size_t> dimSet;
    if (this->dim.size() == 0) {
        for (size_t i = 0; i < inShape.NumAxis(); i++) {
            dimSet.insert(i);
        }
    } else {
        for (auto it : this->dim) {
            dimSet.insert(Operator::GetValidDim(inShape, it));
        }
    }
    return dimSet;
}

Shape ReduceOp::CalculateOutputShape(Shape inputShape, const std::vector<int64_t>& dim, bool keepdim) {
    size_t numAxis = inputShape.NumAxis();

    for (size_t i = 0; i < numAxis; i++) {
        DDebugAssert(inputShape[i] >= 0);
    }

    if (dim.size() == 0) {
        for (size_t i = 0; i < numAxis; i++) {
            inputShape[i] = -1;
        }
    } else {
        for (auto index : dim) {
            if (index < 0) {
                index += static_cast<int>(numAxis);
            }
            if (index < 0 || index >= static_cast<int>(numAxis)) {
                throw std::invalid_argument("Invalid Reduce op dim");
            }

            inputShape[index] = -1;
        }
    }

    Shape outputShape;
    for (size_t i = 0; i < numAxis; i++) {
        if (inputShape[i] == -1) {
            if (keepdim) {
                outputShape.PushBack(1);
            }
        } else {
            outputShape.PushBack(inputShape[i]);
        }
    }

    return outputShape;
}

void ReduceOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);

    const auto& param = GetOpParam<ReduceParam>();
    Shape inputShape = OperandX()->GetShape();
    Shape outputShape;

    outputShape = CalculateOutputShape(inputShape, param.dim, param.keepdim);
    if (param.keepdim && !inputShape.CanBroadcastWith(outputShape)) {
        throw("Invalid reduce operator output shape: " + outputShape.ToString());
    }
    if (inputShape == outputShape) {
        DLogWarning() << "Reduce operator have same input and output size";
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outputShape);

    if (param.dataKind.has_value()) {
        OperandY()->SetDataKind(param.dataKind.value());
    }

    if (param.reduceKind == ReduceKind::kAny || param.reduceKind == ReduceKind::kAll) {
        DDebugAssert(GetInputSize() == 1);
        DDebugAssert(!param.dataKind.has_value());
        // For uint8 the dtype of output is uint8 itself.
        if (OperandY()->GetDataKind() != DataKind::kUInt8) {
            OperandY()->SetDataKind(DataKind::kBool);
        }
    }
}

PlacementSignature ReduceOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    const Shape& inShape = OperandX()->GetShape();
    const auto& param = GetOpParam<ReduceParam>();

    if (param.dim.size() == 0) {
        for (size_t i = 0; i < inShape.NumAxis(); i++) {
            builder.AddInput(Shard(i)).AddOutput(Replicate()).Build();
        }
    } else {
        std::unordered_set<size_t> dimSet = param.GetDimSet(inShape);

        for (size_t inIdx = 0, outIdx = 0; inIdx < inShape.NumAxis(); inIdx++) {
            if (dimSet.find(inIdx) != dimSet.end()) {
                builder.AddInput(Shard(inIdx)).AddOutput(Replicate()).Build();
            } else {
                builder.AddInput(Shard(inIdx)).AddOutput(Shard(outIdx)).Build();
            }

            if (param.keepdim || dimSet.find(inIdx) == dimSet.end()) {
                outIdx++;
            }
        }
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
