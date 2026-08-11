/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

using LinearParam = NoElementOpParam<OperatorType::kLinear>;

class LinearOp : public Operator {
public:
public:
    LinearOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;

    OperatorCost GetOperatorCost() const override;
};

}  // namespace core
}  // namespace dtorch
