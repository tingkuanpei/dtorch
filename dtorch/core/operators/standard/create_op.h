/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <optional>

#include <torch/torch.h>

#include "../operator.h"
#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

// https://pytorch.org/docs/stable/torch.html#creation-ops
// https://pytorch.org/cppdocs/notes/tensor_creation.html#picking-a-factory-function

enum class CreateKind { kZeros = 0, kOnes, kFull, kEye, kEmpty, kArange, kRandInt, kRand, kRandn, kFromTorch, kCount };

struct CreateParam : public OpParam {
    CreateKind createKind;
    Shape shape;
    DataKind dataKind;
    DeviceMesh deviceMesh;
    PlacementSeq placementSeq;
    // rand
    std::optional<Generator> generator;
    // arange full randint
    double doubleArg0;
    double doubleArg1;
    double doubleArg2;
    // set tensor
    std::optional<torch::Tensor> torchValue;

public:
    CreateParam()
        : OpParam(OperatorType::kCreate),
          createKind(CreateKind::kZeros),
          shape(),
          dataKind(DataKind::kFloat32),
          deviceMesh(),
          placementSeq(),
          generator(std::nullopt),
          doubleArg0(0.0),
          doubleArg1(0.0),
          doubleArg2(0.0),
          torchValue(std::nullopt) {}

    CreateParam(CreateKind createKind, const Shape& shape, DataKind dataKind, const DeviceMesh& deviceMesh,
                const PlacementSeq& placementSeq, const std::optional<Generator>& generator = std::nullopt,
                double doubleArg0 = 0.0f, double doubleArg1 = 0.0f, double doubleArg2 = 0.0f,
                const std::optional<torch::Tensor>& torchValue = std::nullopt)
        : OpParam(OperatorType::kCreate),
          createKind(createKind),
          shape(shape),
          dataKind(dataKind),
          deviceMesh(deviceMesh),
          placementSeq(placementSeq),
          generator(generator),
          doubleArg0(doubleArg0),
          doubleArg1(doubleArg1),
          doubleArg2(doubleArg2),
          torchValue(torchValue) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & createKind;
        ar & shape;
        ar & dataKind;
        ar & deviceMesh;
        ar & placementSeq;
        ar & generator;
        ar & doubleArg0;
        ar & doubleArg1;
        ar & doubleArg2;

        // Cpu tensor torchValue too large, we don't serialize it, will copy it to
        // PerDeviceProcessNodeRunner::mCpuRunner directly.
        bool torchValueHasValue = false;
        bool isCpuTensor = false;
        if constexpr (Archive::is_saving::value) {
            torchValueHasValue = torchValue.has_value();
            isCpuTensor = torchValue.has_value() ? torchValue.value().is_cpu() : false;
        }
        ar & isCpuTensor;
        ar & torchValueHasValue;

        if (torchValueHasValue && !isCpuTensor) {
            std::string cudaTensorIpcStr;
            if constexpr (Archive::is_saving::value) {
                cudaTensorIpcStr = external::torch::TorchUtil::ToIpcMemHandle(torchValue.value());
            }
            ar & cudaTensorIpcStr;
            if constexpr (Archive::is_loading::value) {
                // Not clone here, graph.Sync() in dtorch/api/cpp/functional/tensor_functional.cc:_FromTorch() will wait
                // until CreateOp is executed.
                torchValue = external::torch::TorchUtil::FromIpcMemHandle(cudaTensorIpcStr);
            }
        }
    }
};

class CreateOp : public Operator {
public:
public:
    CreateOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

private:
    Shape ComputeShapeForArange(double start, double end, double step) const;
};

}  // namespace core
}  // namespace dtorch
