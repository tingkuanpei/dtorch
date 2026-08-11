/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

enum class NvtxType { kRangePush = 0, kRangePop, kMark, kCount };

struct NvtxParam : public OpParam {
    NvtxType nvtxType;
    std::string message;
    DeviceMesh deviceMesh;

public:
    NvtxParam() : OpParam(OperatorType::kNvtx), nvtxType(NvtxType::kRangePush), message(""), deviceMesh() {}

    NvtxParam(NvtxType nvtxType, const std::string& message, const DeviceMesh& deviceMesh)
        : OpParam(OperatorType::kNvtx), nvtxType(nvtxType), message(message), deviceMesh(deviceMesh) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & nvtxType;
        ar & message;
        ar & deviceMesh;
    }
};

class NvtxOp : public Operator {
public:
public:
    NvtxOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    size_t InferOutputSize() const override { return 0; }

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void InferOperatorAssignInfo() override;

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    // NVTX annotation: no compute, no data moved.
    OperatorCost GetOperatorCost() const override { return OperatorCost{}; }
};

}  // namespace core
}  // namespace dtorch
