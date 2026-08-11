/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct WhereParam : public OpParam {
    std::optional<double> input;
    std::optional<double> other;

public:
    WhereParam(std::optional<double> input = std::nullopt, std::optional<double> other = std::nullopt)
        : OpParam(OperatorType::kWhere), input(input), other(other) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & input;
        ar & other;
    }
};

class WhereOp : public Operator {
public:
public:
    WhereOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    bool IsRequireInputSameDataKind() const override { return false; }

    void InferOutputMetaInfo() const override;

    PlacementSignature GetPlacementSignature() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
