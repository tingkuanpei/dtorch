/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

enum class ActivationType { kReLU = 0, kSigmoid, kLeakyRelu, kELU, kGELU, kSiLU, kCount };

struct ActivationParam : public OpParam {
    ActivationType activationType;
    bool inplace;
    double alpha;
    double beta;
    std::string approximate;

public:
    ActivationParam(ActivationType activationType = ActivationType::kReLU, bool inplace = false, double alpha = 0.0,
                    double beta = 0.0, const std::string& approximate = "")
        : OpParam(OperatorType::kActivation),
          activationType(activationType),
          inplace(inplace),
          alpha(alpha),
          beta(beta),
          approximate(approximate) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & activationType;
        ar & inplace;
        ar & alpha;
        ar & beta;
        ar & approximate;
    }
};

class ActivationOp : public Operator {
public:
public:
    ActivationOp(std::shared_ptr<OpParam> opParam) : Operator(opParam) {}

    void CheckInput() const override;

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
