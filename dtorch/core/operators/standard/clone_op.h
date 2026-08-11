/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

using CloneParam = NoElementOpParam<OperatorType::kClone>;

class CloneOp : public Operator {
public:
public:
    CloneOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
