/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "embedding_op.h"

namespace dtorch {
namespace core {

void EmbeddingOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 2);
    Shape weightShape = OperandWeight()->GetShape();
    if (weightShape.NumAxis() != 2) {
        throw std::invalid_argument("Invalid weight shape: " + weightShape.ToString());
    }
    int64_t embeddingDim = weightShape[1];

    Shape outputShape = OperandX()->GetShape();
    outputShape.PushBack(embeddingDim);
    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetDataKind(OperandWeight()->GetDataKind());
    OperandY()->SetShapeAndStride(outputShape);

    if (!DataKindIsInteger(OperandX()->GetDataKind())) {
        throw std::invalid_argument("Input of Embedding operator should be integer");
    }
}

PlacementSignature EmbeddingOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 2);
    Shape inputShape = OperandX()->GetShape();
    Shape outputShape = OperandY()->GetShape();
    size_t inputDims = inputShape.NumAxis();
    size_t outputDims = outputShape.NumAxis();

    // -------------------------
    // |   X   |  W  |    Y    |
    // -------------------------
    // |   R   | S1  |  S(-1)  |
    // |  SAny |  R  | Same(X) |
    // |   R   | S0  |    R    |
    //
    // W=S0 need communicate, vllm and TensorRT-LLM implement this, for embedding share weight with lm_head

    // Shard + Replicate
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    builder.AddInput("R").AddInput("S1").AddOutput(Shard(outputDims - 1)).Build();
    for (size_t i = 0; i < inputDims; i++) {
        builder.AddInput(Shard(i)).AddInput("R").AddOutput(Shard(i)).Build();
    }

    return builder.Finish();
}

}  // namespace core
}  // namespace dtorch
