/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "../operator.h"

namespace dtorch {
namespace core {

enum class ReduceKind { kSum = 0, kMean, kAny, kAll, kCount };

struct ReduceParam : public OpParam {
    ReduceKind reduceKind;
    std::vector<int64_t> dim;
    bool keepdim;
    std::optional<DataKind> dataKind;

public:
    ReduceParam(ReduceKind reduceKind = ReduceKind::kSum, const IntOrIntArray& dim = {}, bool keepdim = false,
                std::optional<DataKind> dataKind = std::nullopt);

    std::unordered_set<size_t> GetDimSet(const Shape& inShape) const;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & reduceKind;
        ar & dim;
        ar & keepdim;
        ar & dataKind;
    }
};

class ReduceOp : public Operator {
public:
    static Shape CalculateOutputShape(Shape inputShape, const std::vector<int64_t>& axes, bool keepdim);

public:
    ReduceOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    PlacementSignature GetPlacementSignature() const override;
};

}  // namespace core
}  // namespace dtorch
