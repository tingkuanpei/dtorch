/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct ClampParam : public OpParam {
    std::optional<double> min;
    std::optional<double> max;

public:
    ClampParam(const std::optional<double>& min = std::nullopt, const std::optional<double>& max = std::nullopt)
        : OpParam(OperatorType::kClamp), min(min), max(max) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & min;
        ar & max;
    }
};

class ClampOp : public Operator {
public:
public:
    ClampOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    bool IsRequireInputSameDataKind() const override { return false; }
    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
