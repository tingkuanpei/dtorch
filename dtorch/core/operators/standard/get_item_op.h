/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

#include "../operator.h"

namespace dtorch {
namespace core {

struct GetItemParam : public OpParam {
    std::vector<Index> indexVec;

public:
    GetItemParam(std::vector<Index> indexVec = {}) : OpParam(OperatorType::kGetItem), indexVec(std::move(indexVec)) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & indexVec;
    }
};

class GetItemOp : public Operator {
public:
public:
    GetItemOp(std::shared_ptr<OpParam> opParamPtr)
        : Operator(opParamPtr), mIndexVec(), mPlacementSignatureBuilder(), mKeepSubSplitCoordinates(true) {}

    static Shape ComputeIndexedShape(const Shape& inShape, std::vector<Index>& indexVec);

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;

    bool GetKeepSubSplitCoordinates() const noexcept { return mKeepSubSplitCoordinates; }

private:
    mutable std::vector<Index> mIndexVec;
    mutable std::unique_ptr<PlacementSignature::Builder> mPlacementSignatureBuilder;
    // TODO: temp code
    mutable bool mKeepSubSplitCoordinates;
};

}  // namespace core
}  // namespace dtorch
