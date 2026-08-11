/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <stdexcept>

#include "dtorch/core/communication/promise_future/tensor_promise_future.h"
#include "dtorch/core/operators/operator.h"
#include "dtorch/external/boost/boost_serialization.h"

namespace dtorch {
namespace core {

// ============================================================
// GetTensorParam
// ============================================================

struct GetTensorParam : public OpParam {
    // Mutable: ownership is transferred to a background thread in
    // GetTensorOp::Compute() after the CUDA event fires.
    mutable std::unique_ptr<communication::TensorPromise> promise;

    GetTensorParam() : OpParam(OperatorType::kGetTensor), promise(nullptr) {}

    explicit GetTensorParam(std::unique_ptr<communication::TensorPromise> p)
        : OpParam(OperatorType::kGetTensor), promise(std::move(p)) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);

        if constexpr (Archive::is_saving::value) {
            // === Serialize (save) ===
            // Memory mode: promise stays in-process (serialization not used)
            // File mode: serialize type + shared memory filename
            bool hasPromise = (promise != nullptr);
            ar & hasPromise;
            if (hasPromise) {
                communication::TensorPromiseType type = promise->GetType();
                ar & type;
                std::string serializedData = promise->Serialize();
                ar & serializedData;
            }
        } else {
            // === Deserialize (load) ===
            bool hasPromise;
            ar & hasPromise;
            if (hasPromise) {
                communication::TensorPromiseType type;
                ar & type;
                std::string serializedData;
                ar & serializedData;
                // Reconstruct Promise from serialized data via factory
                promise = communication::CreateTensorPromiseFromSerialized(type, serializedData);
            } else {
                promise = nullptr;
            }
        }
    }
};

// ============================================================
// GetTensorOp
// ============================================================

class GetTensorOp : public Operator {
public:
    GetTensorOp(std::shared_ptr<OpParam> opParam) : Operator(opParam) {}

    // Input: target Tensor's Operand
    // Output: 0 (does not produce new Operand)
    size_t InferOutputSize() const override { return 0; }

    void InferOutputMetaInfo() const override {
        // No output, nothing to infer
    }

    void CheckInput() const override {
        // Input tensor must not be a DTensor (GetTensor only gets local tensor value)
        const auto& inputs = GetInputOperands();
        DAlwaysAssert(inputs.size() == 1);
        if (inputs[0]->IsDistributed()) {
            throw std::invalid_argument(
                "GetTensorOp input must be a local tensor, not a DTensor. "
                "Call _Redistribute first with a local DeviceMesh.");
        }
    }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    std::string GetDescribeString() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }
};

}  // namespace core
}  // namespace dtorch
