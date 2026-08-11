/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "broadcast_binary_op.h"

#include <sstream>
#include <unordered_set>

#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

std::string BroadcastBinaryKindToString(BroadcastBinaryKind kind) {
    static const std::array<std::string, 14> kStringMap = {
        "add",           "sub",  "mul",        "div",         "pow",        "equal",   "greater",
        "greater_equal", "less", "less_equal", "logical_and", "logical_or", "minimum", "maximum"};
    static_assert(static_cast<int>(kStringMap.size()) == EnumAsInteger(BroadcastBinaryKind::kCount),
                  "BroadcastBinaryKind size not equal");

    if (kind == BroadcastBinaryKind::kCount) {
        DLogError() << "BroadcastBinaryKind invalid";
        return "";
    }

    return kStringMap[EnumAsInteger(kind)];
}

void BroadcastBinaryOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 2);
    const auto& param = GetOpParam<BroadcastBinaryParam>();
    const Shape& shapeA = OperandA()->GetShape();
    const Shape& shapeB = OperandB()->GetShape();
    Operand* refOperand = OperandA();
    Shape shapeY;

    if (OperandA()->IsNullTensorShape()) {
        DAlwaysAssert(param.inputA.has_value());
        shapeY = shapeB;
        refOperand = OperandB();
    } else if (OperandB()->IsNullTensorShape()) {
        DAlwaysAssert(param.inputB.has_value());
        shapeY = shapeA;
    } else {
        if (!shapeA.CanBroadcastWith(shapeB)) {
            std::stringstream ss;
            ss << "BroadcastBinary op InferOutputShape error, input shape not compatible: ";
            ss << shapeA.ToString() << " vs " << shapeB.ToString();
            throw std::invalid_argument(ss.str());
        }

        shapeY = Shape::BroadcastOutputShape(shapeA, shapeB);
    }

    OperandY()->MetaDataSameAs(refOperand);
    OperandY()->SetShapeAndStride(shapeY);

    // TODO: error implement, need update: https://docs.pytorch.org/docs/stable/tensor_attributes.html#id4
    std::vector<DataKind> inputDataKinds;
    for (size_t i = 0; i < GetInputSize(); i++) {
        // Not null tensor and scalar
        if (!GetInputOperand(i)->IsNullTensorShape() && GetInputOperand(i)->GetShape().NumAxis() != 0) {
            inputDataKinds.push_back(GetInputOperand(i)->GetDataKind());
        }
    }
    if (inputDataKinds.size() > 0) {
        DataKind promotedDataKind = DataKindPromote(inputDataKinds);
        OperandY()->SetDataKind(promotedDataKind);
    }

    std::unordered_set<BroadcastBinaryKind> boolOutputKind = {
        BroadcastBinaryKind::kEqual,    BroadcastBinaryKind::kGreater,   BroadcastBinaryKind::kGreaterEqual,
        BroadcastBinaryKind::kLess,     BroadcastBinaryKind::kLessEqual, BroadcastBinaryKind::kLogicalAnd,
        BroadcastBinaryKind::kLogicalOr};
    if (boolOutputKind.find(param.binaryKind) != boolOutputKind.end()) {
        OperandY()->SetDataKind(DataKind::kBool);
    }
}

std::string BroadcastBinaryOp::GetDescribeString() const {
    const auto& param = GetOpParam<BroadcastBinaryParam>();
    std::stringstream ss;
    ss << GetOpType() << ": " << BroadcastBinaryKindToString(param.binaryKind);
    return ss.str();
}

void BroadcastBinaryOp::AddBroadcastBinaryOpPlacementSignature(PlacementSignature::Builder& builder,
                                                               const Shape& shapeA, const Shape& shapeB) {
    size_t dimA = shapeA.NumAxis();
    size_t dimB = shapeB.NumAxis();
    size_t maxDim = dimA > dimB ? dimA : dimB;
    size_t expandDimA = maxDim - dimA;
    size_t expandDimB = maxDim - dimB;

    // Shard + Replicate
    for (size_t i = 0; i < dimA; i++) {
        if (i < expandDimB || shapeB[i - expandDimB] == 1) {
            builder.AddInput(Shard(i)).AddInput("R").AddOutput(Shard(i + expandDimA)).Build();
        }
    }
    for (size_t i = 0; i < dimB; i++) {
        if (i < expandDimA || shapeA[i - expandDimA] == 1) {
            builder.AddInput("R").AddInput(Shard(i)).AddOutput(Shard(i + expandDimB)).Build();
        }
    }

    for (size_t i = 0; i < dimA && i < dimB; i++) {
        size_t indexA = i + expandDimB;
        size_t indexB = i + expandDimA;
        size_t indexC = i + expandDimA + expandDimB;
        if (shapeA[indexA] == shapeB[indexB]) {
            builder.AddInput(Shard(indexA)).AddInput(Shard(indexB)).AddOutput(Shard(indexC)).Build();
        }
    }
}

PlacementSignature BroadcastBinaryOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 2);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    AddBroadcastBinaryOpPlacementSignature(builder, OperandA()->GetShape(), OperandB()->GetShape());
    return builder.Finish();
}

void BroadcastBinaryOp::Pow(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<BroadcastBinaryParam>();
    torch::Tensor out;

    if (inputs[0].has_value() && inputs[1].has_value()) {
        out = torch::pow(inputs[0].value(), inputs[1].value());
    } else if (inputs[0].has_value() && !inputs[1].has_value()) {
        DDebugAssert(param.inputB.has_value());
        out = torch::pow(inputs[0].value(), external::torch::TorchUtil::ToScalar(param.inputB.value()));
    } else {
        DDebugAssert(param.inputA.has_value());
        out = torch::pow(external::torch::TorchUtil::ToScalar(param.inputA.value()), inputs[1].value());
    }

    outputs.push_back(out);
}

template <typename T>
void BroadcastBinaryOp::AddSubMulDiv(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<BroadcastBinaryParam>();
    torch::Tensor out;

    DDebugAssert(inputs[0].has_value() && !param.inputA.has_value());
    if (param.binaryKind == BroadcastBinaryKind::kMul || param.binaryKind == BroadcastBinaryKind::kDiv) {
        DDebugAssert(!param.scaleB.has_value());
    }

    T scaleB = static_cast<T>(param.scaleB.value_or(1.0));
    const torch::Tensor& inputA = inputs[0].value();
    switch (param.binaryKind) {
        case BroadcastBinaryKind::kAdd:
            if (inputs[1].has_value()) {
                out = inputA.add(inputs[1].value(), scaleB);
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA.add(external::torch::TorchUtil::ToScalar(param.inputB.value()), scaleB);
            }
            break;
        case BroadcastBinaryKind::kSub:
            if (inputs[1].has_value()) {
                out = inputA.sub(inputs[1].value(), scaleB);
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA.sub(external::torch::TorchUtil::ToScalar(param.inputB.value()), scaleB);
            }
            break;
        case BroadcastBinaryKind::kMul:
            if (inputs[1].has_value()) {
                out = inputA.mul(inputs[1].value());
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA.mul(external::torch::TorchUtil::ToScalar(param.inputB.value()));
            }
            break;
        case BroadcastBinaryKind::kDiv:
            if (inputs[1].has_value()) {
                out = inputA.div(inputs[1].value());
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA.div(external::torch::TorchUtil::ToScalar(param.inputB.value()));
            }
            break;
        default:
            DUnimplemented();
            break;
    }

    outputs.push_back(out);
}

void BroadcastBinaryOp::EqualGreaterLess(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<BroadcastBinaryParam>();
    torch::Tensor out;

    DDebugAssert(inputs[0].has_value() && !param.inputA.has_value());
    const torch::Tensor& inputA = inputs[0].value();
    switch (param.binaryKind) {
        case BroadcastBinaryKind::kEqual:
            if (inputs[1].has_value()) {
                out = inputA == inputs[1].value();
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA == external::torch::TorchUtil::ToScalar(param.inputB.value());
            }
            break;
        case BroadcastBinaryKind::kGreater:
            if (inputs[1].has_value()) {
                out = inputA > inputs[1].value();
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA > external::torch::TorchUtil::ToScalar(param.inputB.value());
            }
            break;
        case BroadcastBinaryKind::kGreaterEqual:
            if (inputs[1].has_value()) {
                out = inputA >= inputs[1].value();
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA >= external::torch::TorchUtil::ToScalar(param.inputB.value());
            }
            break;
        case BroadcastBinaryKind::kLess:
            if (inputs[1].has_value()) {
                out = inputA < inputs[1].value();
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA < external::torch::TorchUtil::ToScalar(param.inputB.value());
            }
            break;
        case BroadcastBinaryKind::kLessEqual:
            if (inputs[1].has_value()) {
                out = inputA <= inputs[1].value();
            } else {
                DDebugAssert(param.inputB.has_value());
                out = inputA <= external::torch::TorchUtil::ToScalar(param.inputB.value());
            }
            break;
        default:
            DUnimplemented();
            break;
    }

    outputs.push_back(out);
}

void BroadcastBinaryOp::LogicalAndOr(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<BroadcastBinaryParam>();
    torch::Tensor out;

    DDebugAssert(inputs[0].has_value() && inputs[1].has_value());
    const torch::Tensor& inputA = inputs[0].value();
    const torch::Tensor& inputB = inputs[1].value();
    switch (param.binaryKind) {
        case BroadcastBinaryKind::kLogicalAnd:
            out = torch::logical_and(inputA, inputB);
            break;
        case BroadcastBinaryKind::kLogicalOr:
            out = torch::logical_or(inputA, inputB);
            break;
        default:
            DUnimplemented();
            break;
    }

    outputs.push_back(out);
}

void BroadcastBinaryOp::MinimumMaximum(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<BroadcastBinaryParam>();
    torch::Tensor out;

    DDebugAssert(inputs[0].has_value() && inputs[1].has_value());
    const torch::Tensor& inputA = inputs[0].value();
    const torch::Tensor& inputB = inputs[1].value();
    switch (param.binaryKind) {
        case BroadcastBinaryKind::kMinimum:
            out = torch::minimum(inputA, inputB);
            break;
        case BroadcastBinaryKind::kMaximum:
            out = torch::maximum(inputA, inputB);
            break;
        default:
            DUnimplemented();
            break;
    }

    outputs.push_back(out);
}

void BroadcastBinaryOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 2);
    const auto& param = GetOpParam<BroadcastBinaryParam>();

    if (param.binaryKind == BroadcastBinaryKind::kPow) {
        Pow(inputs, outputs);
    } else if (param.binaryKind == BroadcastBinaryKind::kAdd || param.binaryKind == BroadcastBinaryKind::kSub ||
               param.binaryKind == BroadcastBinaryKind::kMul || param.binaryKind == BroadcastBinaryKind::kDiv) {
        if (DataKindIsInteger(external::torch::TorchUtil::GetDataKind(inputs[0].value()))) {
            AddSubMulDiv<int64_t>(inputs, outputs);
        } else {
            AddSubMulDiv<double>(inputs, outputs);
        }
    } else if (param.binaryKind == BroadcastBinaryKind::kEqual || param.binaryKind == BroadcastBinaryKind::kGreater ||
               param.binaryKind == BroadcastBinaryKind::kGreaterEqual ||
               param.binaryKind == BroadcastBinaryKind::kLess || param.binaryKind == BroadcastBinaryKind::kLessEqual) {
        EqualGreaterLess(inputs, outputs);
    } else if (param.binaryKind == BroadcastBinaryKind::kLogicalAnd ||
               param.binaryKind == BroadcastBinaryKind::kLogicalOr) {
        LogicalAndOr(inputs, outputs);
    } else if (param.binaryKind == BroadcastBinaryKind::kMinimum || param.binaryKind == BroadcastBinaryKind::kMaximum) {
        MinimumMaximum(inputs, outputs);
    } else {
        std::stringstream ss;
        ss << "Unsupport param.binaryKind: " << EnumAsInteger(param.binaryKind);
        throw std::invalid_argument(ss.str());
    }
}

}  // namespace core
}  // namespace dtorch
