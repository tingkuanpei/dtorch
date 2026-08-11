/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct InterpolateParam : public OpParam {
    std::optional<Shape> shape;
    std::optional<std::vector<double>> scaleFactor;
    std::string mode;
    std::optional<bool> alignCorners;
    std::optional<bool> recomputeScaleFactor;
    std::optional<bool> antialias;

public:
    InterpolateParam()
        : OpParam(OperatorType::kInterpolate),
          shape(std::nullopt),
          scaleFactor(std::nullopt),
          mode(""),
          alignCorners(std::nullopt),
          recomputeScaleFactor(std::nullopt),
          antialias(std::nullopt) {}

    InterpolateParam(std::optional<Shape> shape, std::optional<std::vector<double>> scaleFactor, std::string mode,
                     std::optional<bool> alignCorners, std::optional<bool> recomputeScaleFactor,
                     std::optional<bool> antialias)
        : OpParam(OperatorType::kInterpolate),
          shape(shape),
          scaleFactor(scaleFactor),
          mode(mode),
          alignCorners(alignCorners),
          recomputeScaleFactor(recomputeScaleFactor),
          antialias(antialias) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & shape;
        ar & scaleFactor;
        ar & mode;
        ar & alignCorners;
        ar & recomputeScaleFactor;
        ar & antialias;
    }
};

class InterpolateOp : public Operator {
public:
public:
    InterpolateOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
