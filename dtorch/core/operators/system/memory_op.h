/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

enum class MemoryOperationType { kEmptyCache = 0, kMemoryStats, kCount };

struct MemoryParam : public OpParam {
    MemoryOperationType memoryOperationType;
    bool resetPeakStats;
    std::vector<Device> devices;

public:
    MemoryParam()
        : OpParam(OperatorType::kMemory),
          memoryOperationType(MemoryOperationType::kEmptyCache),
          resetPeakStats(false),
          devices() {}

    MemoryParam(MemoryOperationType memoryOperationType, const std::vector<Device>& devices,
                bool resetPeakStats = false)
        : OpParam(OperatorType::kMemory),
          memoryOperationType(memoryOperationType),
          resetPeakStats(resetPeakStats),
          devices(devices) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & memoryOperationType;
        ar & resetPeakStats;
        ar & devices;
    }
};

class MemoryOp : public Operator {
public:
    static constexpr size_t kMemoryStatSize = 256;

    static torch::Tensor ToTorchTensor(const MemoryStats& memoryStats);

    static MemoryStats ToMemoryStats(const torch::Tensor& tensor);

public:
    MemoryOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    size_t InferOutputSize() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void InferOperatorAssignInfo() override;

    // Introspection / cache control: not a tensor-data mover.
    OperatorCost GetOperatorCost() const override { return OperatorCost{}; }
};

}  // namespace core
}  // namespace dtorch
