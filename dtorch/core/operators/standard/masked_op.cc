/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "masked_op.h"

#include <sstream>
#include <stdexcept>

#include "dtorch/common/debug.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

// MaskedOp — Masked Fill / Masked Scatter
//
// == 功能描述 (Functionality) ==
// 实现 PyTorch 的 masked_fill, masked_scatter 语义。
//
// masked_fill (kMaskedFill):
//   将 input 中 mask=True 的位置填充为标量 value。
//   等价于: output[i] = mask[i] ? value : input[i]
//   对应: torch.masked_fill(input, mask, value)
//   - input: 任意 shape 的 Tensor
//   - mask: BoolTensor，broadcastable with input
//   - value: 标量填充值
//
// masked_scatter (kMaskedScatter):
//   将 source 中的元素按行优先顺序依次拷贝到 input 中 mask=True 的位置。
//   对应: torch.masked_scatter(input, mask, source)
//   - input: 任意 shape 的 Tensor
//   - mask: BoolTensor，broadcastable with input
//   - source: 任意 shape 的 Tensor，提供填充元素。source.numel() >= mask.sum()
//
// == 输出 Shape (Output Shape) ==
// masked_fill / masked_scatter:
//   输出 Shape 为 input 与 mask broadcast 后的 Shape，与 input 相同 dtype。
//   推导步骤: outputShape = BroadcastOutputShape(inputShape, maskShape)
//
// == Placements 推导 (PlacementSignature) ==
// masked_fill / masked_scatter: 输出 Placement 与输入相同（element-wise 语义）。
// 通过 SkipDistributedSpecFromPlacementSignature() 返回 true 跳过分布式推断。

void MaskedOp::InferOutputMetaInfo() const {
    const auto& param = GetOpParam<MaskedParam>();

    switch (param.maskedType) {
        case MaskedType::kMaskedFill: {
            // inputs: input (OperandA), mask (OperandB); param: value (scalar)
            DDebugAssert(GetInputSize() == 2);
            const Shape& inputShape = OperandA()->GetShape();
            const Shape& maskShape = OperandB()->GetShape();

            if (OperandB()->GetDataKind() != DataKind::kBool) {
                std::stringstream ss;
                ss << "masked_fill expected mask to be a boolean tensor, but got a tensor with dtype "
                   << OperandB()->GetDataKind();
                throw std::invalid_argument(ss.str());
            }

            Shape outputShape = inputShape;
            if (!outputShape.CanBroadcastWith(maskShape)) {
                std::stringstream ss;
                ss << "MaskedOp(masked_fill) InferOutputShape error, shape not compatible: ";
                ss << outputShape.ToString() << " vs " << maskShape.ToString();
                throw std::invalid_argument(ss.str());
            }
            outputShape = Shape::BroadcastOutputShape(outputShape, maskShape);

            OperandY()->MetaDataSameAs(OperandX());
            OperandY()->SetShapeAndStride(outputShape);
            OperandY()->SetDataKind(OperandA()->GetDataKind());
            break;
        }
        case MaskedType::kMaskedScatter: {
            // inputs: input (OperandA), mask (OperandB), source (OperandC)
            DDebugAssert(GetInputSize() == 3);
            const Shape& inputShape = OperandA()->GetShape();
            const Shape& maskShape = OperandB()->GetShape();

            if (OperandB()->GetDataKind() != DataKind::kBool) {
                std::stringstream ss;
                ss << "masked_scatter expected mask to be a boolean tensor, but got a tensor with dtype "
                   << OperandB()->GetDataKind();
                throw std::invalid_argument(ss.str());
            }

            Shape outputShape = inputShape;
            if (!outputShape.CanBroadcastWith(maskShape)) {
                std::stringstream ss;
                ss << "MaskedOp(masked_scatter) InferOutputShape error, shape not compatible: ";
                ss << outputShape.ToString() << " vs " << maskShape.ToString();
                throw std::invalid_argument(ss.str());
            }
            outputShape = Shape::BroadcastOutputShape(outputShape, maskShape);

            OperandY()->MetaDataSameAs(OperandX());
            OperandY()->SetShapeAndStride(outputShape);
            OperandY()->SetDataKind(OperandA()->GetDataKind());
            break;
        }
    }
}

bool MaskedOp::SkipDistributedSpecFromPlacementSignature() const {
    // masked_fill and masked_scatter produce output with same shape as input (after broadcast),
    // so output placement can mirror the first input's placement.
    return true;
}

PlacementSignature MaskedOp::GetPlacementSignature() const {
    const auto& param = GetOpParam<MaskedParam>();

    size_t inputSize = 0;
    switch (param.maskedType) {
        case MaskedType::kMaskedFill:
            inputSize = 2;
            break;
        case MaskedType::kMaskedScatter:
            inputSize = 3;
            break;
    }

    PlacementSignature::Builder builder(inputSize, GetOutputSize());
    // TODO: Define proper placement propagation rules for masked operations.
    // - masked_fill / masked_scatter: output same placement as input (element-wise).

    return builder.Finish();
}

std::string MaskedOp::GetDescribeString() const {
    const auto& param = GetOpParam<MaskedParam>();
    std::stringstream ss;
    switch (param.maskedType) {
        case MaskedType::kMaskedFill:
            ss << "MaskedOp(masked_fill, value=" << param.value << ")";
            break;
        case MaskedType::kMaskedScatter:
            ss << "MaskedOp(masked_scatter)";
            break;
    }
    return ss.str();
}

}  // namespace core
}  // namespace dtorch
