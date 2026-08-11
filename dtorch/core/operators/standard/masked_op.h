/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

enum class MaskedType { kMaskedFill, kMaskedScatter };

struct MaskedParam : public OpParam {
    MaskedType maskedType;
    double value;  // only used by kMaskedFill

public:
    MaskedParam(MaskedType maskedType = MaskedType::kMaskedFill, double value = 0.0)
        : OpParam(OperatorType::kMasked), maskedType(maskedType), value(value) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & maskedType;
        ar & value;
    }
};

class MaskedOp : public Operator {
public:
public:
    MaskedOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    bool IsRequireInputSameDataKind() const override { return false; }

    bool SkipDistributedSpecFromPlacementSignature() const override;

    void InferOutputMetaInfo() const override;

    PlacementSignature GetPlacementSignature() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    std::string GetDescribeString() const override;
};

}  // namespace core
}  // namespace dtorch
