/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "../operator.h"

namespace dtorch {
namespace core {

enum class BroadcastBinaryKind {
    kAdd = 0,
    kSub,
    kMul,
    kDiv,
    kPow,
    kEqual,
    kGreater,
    kGreaterEqual,
    kLess,
    kLessEqual,
    kLogicalAnd,
    kLogicalOr,
    kMinimum,
    kMaximum,
    kCount
};

std::string BroadcastBinaryKindToString(BroadcastBinaryKind kind);

struct BroadcastBinaryParam : public OpParam {
    BroadcastBinaryKind binaryKind;
    std::optional<Scalar> inputA;
    std::optional<Scalar> inputB;
    std::optional<double> scaleB;

public:
    BroadcastBinaryParam(BroadcastBinaryKind binaryKind = BroadcastBinaryKind::kAdd,
                         const std::optional<Scalar>& inputA = std::nullopt,
                         const std::optional<Scalar>& inputB = std::nullopt,
                         const std::optional<double>& scaleB = std::nullopt)
        : OpParam(OperatorType::kBroadcastBinary),
          binaryKind(binaryKind),
          inputA(inputA),
          inputB(inputB),
          scaleB(scaleB) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & binaryKind;
        ar & inputA;
        ar & inputB;
        ar & scaleB;
    }
};

class BroadcastBinaryOp : public Operator {
public:
public:
    BroadcastBinaryOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    bool IsRequireInputSameDataKind() const override { return false; }

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    std::string GetDescribeString() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override {
        return !OperandA()->IsDistributed() || OperandB()->IsNullTensorShape() || OperandA()->IsNullTensorShape();
    }

    PlacementSignature GetPlacementSignature() const override;

private:
    void Pow(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const;

    template <typename T>
    void AddSubMulDiv(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const;

    void EqualGreaterLess(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const;

    void LogicalAndOr(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const;

    void MinimumMaximum(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const;

public:
    static void AddBroadcastBinaryOpPlacementSignature(PlacementSignature::Builder& builder, const Shape& shapeA,
                                                       const Shape& shapeB);
};

}  // namespace core
}  // namespace dtorch
