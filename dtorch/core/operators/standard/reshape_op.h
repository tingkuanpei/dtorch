/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct ReshapeParam : public OpParam {
    Shape shape;

public:
    ReshapeParam(Shape shape = {}) : OpParam(OperatorType::kReshape), shape(shape) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & shape;
    }
};

class ReshapeOp : public Operator {
public:
    static bool CalculateOutputShape(const Shape& inputShape, const Shape& expectShape, Shape& outputShape);

public:
    ReshapeOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    std::string GetDescribeString() const override;

    PlacementSignature GetPlacementSignature() const override;
};

}  // namespace core
}  // namespace dtorch
