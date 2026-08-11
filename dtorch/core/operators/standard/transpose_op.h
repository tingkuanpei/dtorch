/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct TransposeParam : public OpParam {
    int64_t dim0;
    int64_t dim1;

public:
    TransposeParam(int64_t dim0 = 0, int64_t dim1 = 1) : OpParam(OperatorType::kTranspose), dim0(dim0), dim1(dim1) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & dim0;
        ar & dim1;
    }
};

class TransposeOp : public Operator {
public:
public:
    TransposeOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;
};

}  // namespace core
}  // namespace dtorch
