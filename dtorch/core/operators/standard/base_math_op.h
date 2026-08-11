/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

enum class BaseMathType {
    kExp = 0,
    kSquare,
    kRsqrt,
    kAbs,
    kRound,
    kFloor,
    kCos,
    kSin,
    kAsin,
    kTanh,
    kNeg,
    kReciprocal,
    kLog,
    kLog2,
    kLog10,
    kIsInf,
    kIsNan,
    kCount
};

struct BaseMathParam : public OpParam {
    BaseMathType baseMathType;

public:
    BaseMathParam(BaseMathType baseMathType = BaseMathType::kExp)
        : OpParam(OperatorType::kBaseMath), baseMathType(baseMathType) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & baseMathType;
    }
};

class BaseMathOp : public Operator {
public:
public:
    BaseMathOp(std::shared_ptr<OpParam> opParam) : Operator(opParam) {}

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
