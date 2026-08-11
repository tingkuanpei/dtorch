/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"
#include "dtorch/api/cpp/int_or_int_array.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

enum class NormalizationKind { kLayerNorm = 0, kRmsNorm, kGroupNorm, kCount };

struct NormalizationParam : public OpParam {
    NormalizationKind normKind;
    Shape normalizedShape;
    double epsilon;
    int64_t numGroups;

public:
    NormalizationParam()
        : OpParam(OperatorType::kNormalization),
          normKind(NormalizationKind::kLayerNorm),
          normalizedShape(),
          epsilon(1e-5),
          numGroups(0) {}

    NormalizationParam(NormalizationKind normKind, const Shape& normalizedShape, double epsilon = 1e-5,
                       int64_t numGroups = 0)
        : OpParam(OperatorType::kNormalization),
          normKind(normKind),
          normalizedShape(normalizedShape),
          epsilon(epsilon),
          numGroups(numGroups) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & normKind;
        ar & normalizedShape;
        ar & epsilon;
        ar & numGroups;
    }
};

class NormalizationOp : public Operator {
public:
public:
    NormalizationOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    PlacementSignature GetPlacementSignature() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
