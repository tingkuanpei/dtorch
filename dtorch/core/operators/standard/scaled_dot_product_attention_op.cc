/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "scaled_dot_product_attention_op.h"

#include <torch/torch.h>

#include "dtorch/external/sage_attn/sage_attn_adapter.h"

namespace dtorch {
namespace core {

void SdpaOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 4);
    const Shape& qShape = OperandA()->GetShape();
    const Shape& kShape = OperandB()->GetShape();
    const Shape& vShape = OperandC()->GetShape();

    DDebugAssert(qShape.NumAxis() >= 4);
    DDebugAssert(kShape.NumAxis() >= 4);
    DDebugAssert(vShape.NumAxis() >= 4);
    // https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    Shape::DataType E_v = vShape[-1];
    Shape outShape = qShape;
    outShape[-1] = E_v;

    OperandY()->MetaDataSameAs(OperandA());
    OperandY()->SetShapeAndStride(outShape);
}

PlacementSignature SdpaOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 4);
    size_t numAxis = OperandA()->GetShape().NumAxis();
    DDebugAssert(numAxis >= 4);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    int64_t shardIndex = 0;
    builder.AddInput(Shard(shardIndex))
        .AddInput(Shard(shardIndex))
        .AddInput(Shard(shardIndex))
        .AddOptionalInput(Shard(shardIndex))
        .AddOutput(Shard(shardIndex))
        .Build();

    for (size_t i = 1; i < numAxis - 2; i++) {
        builder.AddInput(Shard(i))
            .AddInput(Shard(i))
            .AddInput(Shard(i))
            .AddOptionalInput("R")
            .AddOutput(Shard(i))
            .Build();
    }

    shardIndex = 2;
    builder.AddInput(Shard(shardIndex))
        .AddInput("R")
        .AddInput("R")
        .AddOptionalInput("R")
        .AddOutput(Shard(shardIndex))
        .Build();

    return builder.Finish();
}

// FLOPs derivation for scaled-dot-product attention.
//
// All tensors share layout [...batch, heads, seq, head_dim / E_v]. For one (batch, qHead) pair the
// attention is two batched GEMMs:
//   1. scores = Q @ K^T : [seqQ, headDim] × [headDim, seqKv] → [seqQ, seqKv], 2 * seqQ * seqKv * headDim
//   2. out    = p @ V   : [seqQ, seqKv]  × [seqKv,  E_v]    → [seqQ, E_v],  2 * seqQ * seqKv * E_v
// Scaling both by (batch * heads) and summing gives 2 * batch * heads * seqQ * seqKv * (headDim + E_v).
// headDim == kShape[-1] (the Q/K contraction) and E_v == vShape[-1] (the V contraction; may differ).
//
// Grouped-query attention (GQA): K/V carry fewer heads than Q but are broadcast/replicated across Q
// heads, so the work still scales with the Q head count — use qHeads regardless of enableGqa.
// Softmax/exp is omitted (memory-bound relative to the two GEMMs).
OperatorCost SdpaOp::GetOperatorCost() const {
    DDebugAssert(GetInputSize() == 4);
    const Shape& qShape = OperandA()->GetShape();
    const Shape& kShape = OperandB()->GetShape();
    const Shape& vShape = OperandC()->GetShape();

    // Layout: [ ...batch, heads, seq, head_dim / E_v ]
    int64_t batch = static_cast<int64_t>(qShape.Prod(0, static_cast<int>(qShape.NumAxis()) - 4));
    int64_t qHeads = static_cast<int64_t>(qShape[-3]);
    int64_t seqQ = static_cast<int64_t>(qShape[-2]);
    int64_t headDim = static_cast<int64_t>(qShape[-1]);  // == kShape[-1]
    int64_t seqKv = static_cast<int64_t>(kShape[-2]);
    int64_t eV = static_cast<int64_t>(vShape[-1]);

    return OperatorCost::Compute(2 * batch * qHeads * seqQ * seqKv * (headDim + eV));
}

void SdpaOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
#if !DTORCH_INTEL_MAXOS_TORCH_2_2_2
    const auto& param = GetOpParam<SdpaParam>();
    DDebugAssert(inputs.size() == 4);
    std::string sageAttentionType = "";
    if (param.option.has_value()) {
        sageAttentionType = param.option.value().sageAttentionType;
    }

    if (sageAttentionType == "") {
        outputs.push_back(torch::scaled_dot_product_attention(inputs[0].value(), inputs[1].value(), inputs[2].value(),
                                                              inputs[3], 0.0, param.isCausal, param.scale,
                                                              param.enableGqa));
    } else {
        std::optional<float> smScale = std::nullopt;
        if (param.scale.has_value()) {
            smScale = param.scale.value();
        }
        outputs.push_back(external::sage::SageAttnAdapter(inputs[0].value(), inputs[1].value(), inputs[2].value(),
                                                          param.isCausal, smScale, sageAttentionType));
    }
#else
    IgnoreUnused(inputs, outputs);
    DUnimplemented();
#endif
}

}  // namespace core
}  // namespace dtorch
