/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct LayerNormMulAddParam : public OpParam {
    Shape normalizedShape;
    double epsilon;

public:
    LayerNormMulAddParam(Shape normalizedShape = {}, double epsilon = 1e-5)
        : OpParam(OperatorType::kLayerNormMulAdd), normalizedShape(normalizedShape), epsilon(epsilon) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & normalizedShape;
        ar & epsilon;
    }
};

class LayerNormMulAddOp : public Operator {
public:
public:
    LayerNormMulAddOp(std::shared_ptr<OpParam> opParam) : Operator(opParam) {}

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
