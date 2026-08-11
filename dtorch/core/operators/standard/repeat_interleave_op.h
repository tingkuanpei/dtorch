/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "../operator.h"

namespace dtorch {
namespace core {

struct RepeatInterleaveParam : public OpParam {
    int64_t repeats;
    std::optional<int64_t> dim;

public:
    RepeatInterleaveParam(int64_t repeats = 1, std::optional<int64_t> dim = std::nullopt)
        : OpParam(OperatorType::kRepeatInterleave), repeats(repeats), dim(dim) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & repeats;
        ar & dim;
    }
};

class RepeatInterleaveOp : public Operator {
public:
public:
    RepeatInterleaveOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;
};

}  // namespace core
}  // namespace dtorch
