/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct ExpandParam : public OpParam {
    Shape shape;

public:
    ExpandParam() : OpParam(OperatorType::kExpand), shape() {}

    ExpandParam(const Shape& shape) : OpParam(OperatorType::kExpand), shape(shape) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & shape;
    }
};

class ExpandOp : public Operator {
public:
public:
    ExpandOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;
};

}  // namespace core
}  // namespace dtorch
