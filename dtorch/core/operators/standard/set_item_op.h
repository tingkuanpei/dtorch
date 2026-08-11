/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

#include "../operator.h"

namespace dtorch {
namespace core {

struct SetItemParam : public OpParam {
    std::vector<Index> indexVec;

public:
    SetItemParam(std::vector<Index> indexVec = {}) : OpParam(OperatorType::kSetItem), indexVec(std::move(indexVec)) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & indexVec;
    }
};

class SetItemOp : public Operator {
public:
public:
    SetItemOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr), mIndexVec() {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

private:
    mutable std::vector<Index> mIndexVec;
};

}  // namespace core
}  // namespace dtorch
