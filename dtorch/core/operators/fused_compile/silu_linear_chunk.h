/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct SiluLinearChunkParam : public OpParam {
    int64_t chunkSize;

public:
    SiluLinearChunkParam(int64_t chunkSize = 0) : OpParam(OperatorType::kSiluLinearChunk), chunkSize(chunkSize) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & chunkSize;
    }
};

class SiluLinearChunkOp : public Operator {
public:
public:
    SiluLinearChunkOp(std::shared_ptr<OpParam> opParam) : Operator(opParam) {}

    size_t InferOutputSize() const override;

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
