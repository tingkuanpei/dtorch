/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct PermuteParam : public OpParam {
    std::vector<int64_t> dims;

public:
    PermuteParam(const std::vector<int64_t>& dims = {}) : OpParam(OperatorType::kPermute), dims(dims) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & dims;
    }
};

class PermuteOp : public Operator {
public:
public:
    PermuteOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;
};

}  // namespace core
}  // namespace dtorch
