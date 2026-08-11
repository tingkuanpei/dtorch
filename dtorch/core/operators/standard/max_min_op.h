/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "../operator.h"

namespace dtorch {
namespace core {

enum class MaxMinKind { kMax = 0, kMin, kCount };

struct MaxMinParam : public OpParam {
    MaxMinKind maxMinKind;
    std::optional<int64_t> dim;
    bool keepdim;

public:
    MaxMinParam(MaxMinKind maxMinKind = MaxMinKind::kMax)
        : OpParam(OperatorType::kMaxMin), maxMinKind(maxMinKind), dim(std::nullopt), keepdim(false) {}

    MaxMinParam(MaxMinKind maxMinKind, int64_t dim, bool keepdim)
        : OpParam(OperatorType::kMaxMin), maxMinKind(maxMinKind), dim(dim), keepdim(keepdim) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & maxMinKind;
        ar & dim;
        ar & keepdim;
    }
};

class MaxMinOp : public Operator {
public:
public:
    MaxMinOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    size_t InferOutputSize() const override;

    void InferOutputMetaInfo() const override;

    PlacementSignature GetPlacementSignature() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
