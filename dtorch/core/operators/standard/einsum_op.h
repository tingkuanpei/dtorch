/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct EinsumParam : public OpParam {
    std::string equation;

public:
    EinsumParam(const std::string& equation = "") : OpParam(OperatorType::kEinsum), equation(equation) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & equation;
    }
};

class EinsumOp : public Operator {
public:
public:
    EinsumOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr), mPlacementSignatureBuilder() {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;

private:
    mutable std::unique_ptr<PlacementSignature::Builder> mPlacementSignatureBuilder;
};

}  // namespace core
}  // namespace dtorch
