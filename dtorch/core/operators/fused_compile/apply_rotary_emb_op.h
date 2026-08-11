/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct ApplyRotaryEmbParam : public OpParam {
    bool useReal;
    int64_t useRealUnbindDim;

public:
    ApplyRotaryEmbParam(bool useReal = true, int64_t useRealUnbindDim = -1)
        : OpParam(OperatorType::kApplyRotaryEmb), useReal(useReal), useRealUnbindDim(useRealUnbindDim) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & useReal;
        ar & useRealUnbindDim;
    }
};

class ApplyRotaryEmbOp : public Operator {
public:
public:
    ApplyRotaryEmbOp(std::shared_ptr<OpParam> opParam) : Operator(opParam) {}

    bool IsRequireInputSameDataKind() const override { return false; }

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
