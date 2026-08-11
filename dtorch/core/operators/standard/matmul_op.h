/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

using MatmulParam = NoElementOpParam<OperatorType::kMatmul>;

class MatmulOp : public Operator {
public:
public:
    MatmulOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void CheckInput() const override;

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    std::string GetDescribeString() const override;

    OperatorCost GetOperatorCost() const override;

private:
    Shape ComputeShapeForLessTwoDim(const Shape& shapeA, const Shape& shapeB) const;

    std::tuple<Shape, Shape> SplitBroadcastShape(const Shape& shape) const;

    Shape MergeShape(const Shape& shapeA, const Shape& shapeB) const;

    PlacementSignature GetPlacementSignature() const override;
};

}  // namespace core
}  // namespace dtorch
