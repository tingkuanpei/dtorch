/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct FlattenParam : public OpParam {
    int64_t startDim;
    int64_t endDim;

public:
    FlattenParam(int64_t startDim = 0, int64_t endDim = -1)
        : OpParam(OperatorType::kFlatten), startDim(startDim), endDim(endDim) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & startDim;
        ar & endDim;
    }
};

class FlattenOp : public Operator {
public:
    static bool CalculateOutputShape(const Shape& inputShape, const FlattenParam& param, Shape& outputShape);

public:
    FlattenOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    std::string GetDescribeString() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;
};

}  // namespace core
}  // namespace dtorch
