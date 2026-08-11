/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

using EmbeddingParam = NoElementOpParam<OperatorType::kEmbedding>;

class EmbeddingOp : public Operator {
public:
public:
    EmbeddingOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    bool IsRequireInputSameDataKind() const override { return false; }

    void InferOutputMetaInfo() const override;

    PlacementSignature GetPlacementSignature() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
