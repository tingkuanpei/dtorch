/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "outer_op.h"

namespace dtorch {
namespace core {

void OuterOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 2);
    const Shape& shapeA = OperandA()->GetShape();
    const Shape& shapeB = OperandB()->GetShape();

    if (shapeA.NumAxis() != 1) {
        std::stringstream ss;
        ss << "outer: Expected 1-D argument input, but got " << shapeA.NumAxis() << "-D";
        throw std::invalid_argument(ss.str());
    }

    if (shapeB.NumAxis() != 1) {
        std::stringstream ss;
        ss << "outer: Expected 1-D argument vec2, but got " << shapeB.NumAxis() << "-D";
        throw std::invalid_argument(ss.str());
    }

    Shape outShape = {shapeA[0], shapeB[0]};

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outShape);

    std::vector<DataKind> inputDataKinds = {OperandA()->GetDataKind(), OperandB()->GetDataKind()};
    if (inputDataKinds.size() > 0) {
        DataKind promotedDataKind = DataKindPromote(inputDataKinds);
        OperandY()->SetDataKind(promotedDataKind);
    }
}

PlacementSignature OuterOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 2);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("P").AddInput(Replicate()).AddOutput("P").Build();
    builder.AddInput(Replicate()).AddInput("P").AddOutput("P").Build();
    builder.AddInput(Shard(0)).AddInput(Replicate()).AddOutput(Shard(0)).Build();
    builder.AddInput(Replicate()).AddInput(Shard(0)).AddOutput(Shard(1)).Build();

    return builder.Finish();
}

// FLOPs derivation for the outer product of two 1-D vectors.
//
// out[i, j] = a[i] * b[j] has numelA * numelB elements, each a single multiply with no reduction.
// That is 1 FLOP per output element — NOT the 2 FLOPs/MAC convention used for matmul/conv/linear,
// which contract over an extra K axis. Hence numelA * numelB with no leading factor of 2.
OperatorCost OuterOp::GetOperatorCost() const {
    DDebugAssert(GetInputSize() == 2);
    const Shape& shapeA = OperandA()->GetShape();
    const Shape& shapeB = OperandB()->GetShape();
    DDebugAssert(shapeA.NumAxis() == 1);
    DDebugAssert(shapeB.NumAxis() == 1);

    int64_t numelA = static_cast<int64_t>(shapeA[0]);
    int64_t numelB = static_cast<int64_t>(shapeB[0]);

    return OperatorCost::Compute(numelA * numelB);
}

}  // namespace core
}  // namespace dtorch
