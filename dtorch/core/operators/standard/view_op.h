/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "../operator.h"

namespace dtorch {
namespace core {

struct ViewParam : public OpParam {
    std::optional<Shape> shape;
    std::optional<PlacementSeq> placements;

public:
    ViewParam(Shape shape = {}) : OpParam(OperatorType::kView), shape(shape), placements(std::nullopt) {}

    ViewParam(PlacementSeq placements) : OpParam(OperatorType::kView), shape(std::nullopt), placements(placements) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & shape;
        ar & placements;
    }
};

class ViewOp : public Operator {
public:
public:
    ViewOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override {
        const auto& param = GetOpParam<ViewParam>();
        return param.placements.has_value() || Operator::SkipDistributedSpecFromPlacementSignature();
    }

    PlacementSignature GetPlacementSignature() const override;

    std::string GetDescribeString() const override;
};

}  // namespace core
}  // namespace dtorch
