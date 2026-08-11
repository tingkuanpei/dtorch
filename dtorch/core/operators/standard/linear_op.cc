/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "linear_op.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void LinearOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 3);

    Operand* inputA = OperandX();
    Operand* inputB = OperandWeight();
    const Shape& inputAShape = inputA->GetShape();
    const Shape& inputBShape = inputB->GetShape();
    if (inputBShape.NumAxis() != 2) {
        std::stringstream ss;
        ss << "Weight of Linear MUST have 2 dimension: " << inputBShape;
        throw std::invalid_argument(ss.str());
    }

    Shape::DataType inFeatures, outFeatures;
    inFeatures = inputAShape[-1];
    outFeatures = inputBShape[0];
    DDebugAssert(inFeatures == inputBShape[1]);

    Shape outputShape = inputAShape;
    outputShape[-1] = outFeatures;

    if (!OperandBias()->IsNullTensorShape()) {
        const Shape& inputCShape = OperandBias()->GetShape();
        if (inputCShape.NumAxis() != 1 && inputCShape[0] != outFeatures) {
            throw std::invalid_argument("Linear op InferOutputShape error, unacceptable input C: " +
                                        inputCShape.ToString());
        }
    }
    OperandY()->MetaDataSameAs(inputA);
    OperandY()->SetShapeAndStride(outputShape);
}

PlacementSignature LinearOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 3);

    // GEMM:
    // ---------------------
    // |  X  |  W  |  Y  |
    // ---------------------
    // | S0  |  R  | S0  |
    // |  R  | S1  | S1  |
    // | S1  | S0  |  P  |
    // |  P  |  R  |  P  |
    // |  R  |  P  |  P  |

    // Linear: W is transpose, B should same as Y
    // ---------------------
    // |  X  |  W  |  B  |  Y  |
    // ---------------------
    // | S0  |  R  |  R  | S0  |
    // |  R  | S0  | S0  | S1  |
    // | S1  | S1  |  P  |  P  |
    // |  P  |  R  |  P  |  P  |
    // |  R  |  P  |  P  |  P  |
    //
    // Shape of X is (∗,in_features) where * means any number of additional, so
    // S0 of X, Y meaning shared all dim expect -1
    // S1 of X, Y meaning shared at -1

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());

    const Shape& inputAShape = OperandX()->GetShape();
    DDebugAssert(inputAShape.NumAxis() >= 2);
    for (size_t i = 0; i < inputAShape.NumAxis() - 1; i++) {
        builder.AddInput(Shard(i)).AddInput("R").AddOptionalInput("R").AddOutput(Shard(i)).Build();
    }
    builder.AddInput("R").AddInput("S0").AddOptionalInput("S0").AddOutput(Shard(inputAShape.NumAxis() - 1)).Build();
    builder.AddInput(Shard(inputAShape.NumAxis() - 1)).AddInput("S1").AddOptionalInput("P").AddOutput("P").Build();
    builder.AddInput("P").AddInput("R").AddOptionalInput("P").AddOutput("P").Build();
    builder.AddInput("R").AddInput("P").AddOptionalInput("P").AddOutput("P").Build();

    return builder.Finish();
}

// FLOPs derivation for nn.Linear (a batched GEMM against a 2-D weight).
//
// Input X has shape [*, inFeatures] (PyTorch Linear convention: '*' is any number of batch dims);
// weight W has shape [outFeatures, inFeatures]. Linear folds every leading batch dim into a single
// `batchSize = Prod(0, numAxis-2)` (Shape::Prod is inclusive on both ends), then the contraction is a
// plain GEMM of `batchSize` rows: each of the (batchSize * outFeatures) outputs sums inFeatures MACs.
// At 2 FLOPs per MAC that is 2 * batchSize * inFeatures * outFeatures.
// The trailing bias add is elementwise (memory-bound) and is intentionally not counted here.
OperatorCost LinearOp::GetOperatorCost() const {
    DDebugAssert(GetInputSize() == 3);
    const Shape& inputShape = OperandX()->GetShape();
    const Shape& weightShape = OperandWeight()->GetShape();
    DDebugAssert(weightShape.NumAxis() == 2);

    size_t numAxis = inputShape.NumAxis();
    int64_t batchSize = (numAxis >= 2) ? static_cast<int64_t>(inputShape.Prod(0, static_cast<int>(numAxis) - 2)) : 1;
    int64_t inFeatures = static_cast<int64_t>(inputShape[numAxis - 1]);
    int64_t outFeatures = static_cast<int64_t>(weightShape[0]);

    return OperatorCost::Compute(2 * batchSize * inFeatures * outFeatures);
}

}  // namespace core
}  // namespace dtorch
