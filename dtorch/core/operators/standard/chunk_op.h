/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

struct ChunkParam : public OpParam {
    int64_t chunks;
    int64_t dim;

public:
    ChunkParam(int64_t chunks = 1, int64_t dim = 0) : OpParam(OperatorType::kChunk), chunks(chunks), dim(dim) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & chunks;
        ar & dim;
    }
};

class ChunkOp : public Operator {
public:
public:
    ChunkOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    size_t InferOutputSize() const override;

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;
};

}  // namespace core
}  // namespace dtorch
