/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"
#include "dtorch/api/cpp/int_or_int_array.h"

namespace dtorch {
namespace core {

struct SqueezeParam : public OpParam {
    std::vector<int64_t> dims;

public:
    SqueezeParam(const std::optional<IntOrIntArray>& optDims = std::nullopt) : OpParam(OperatorType::kSqueeze), dims() {
        if (optDims.has_value()) {
            dims = optDims.value().Vec();
        }
    }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & dims;
    }
};

class SqueezeOp : public Operator {
public:
public:
    SqueezeOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr), mIODimMap() {}

    void InferOutputMetaInfo() const override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    PlacementSignature GetPlacementSignature() const override;

    std::string GetDescribeString() const override;

private:
    mutable std::unordered_map<size_t, size_t> mIODimMap;
};

}  // namespace core
}  // namespace dtorch
