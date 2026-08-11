/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <vector>

#include "dtorch/core/communication/promise_future/void_promise_future.h"
#include "dtorch/core/operators/operator.h"
#include "dtorch/external/boost/boost_serialization.h"

namespace dtorch {
namespace core {

// ============================================================
// SyncParam
// ============================================================
//
// Holds the devices to sync and one VoidPromise per device.
// One SyncOp handles all devices: InferOperatorAssignInfo() creates
// one stream key per device, so each device gets its own SyncKernel.
// Each SyncKernel fulfills the VoidPromise for its device.

struct SyncParam : public OpParam {
    // One promise per device. Mutable: ownership of individual promises
    // is transferred to SyncKernel which calls SetValue().
    mutable std::vector<std::unique_ptr<communication::VoidPromise>> promises;
    std::vector<Device> syncDevices;

    SyncParam() : OpParam(OperatorType::kSync), promises(), syncDevices() {}

    SyncParam(std::vector<Device> devices, std::vector<std::unique_ptr<communication::VoidPromise>> p)
        : OpParam(OperatorType::kSync), promises(std::move(p)), syncDevices(std::move(devices)) {
        DAlwaysAssert(syncDevices.size() == promises.size());
    }

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & syncDevices;

        if constexpr (Archive::is_saving::value) {
            size_t count = promises.size();
            ar & count;
            for (auto& p : promises) {
                communication::VoidPromiseType type = p->GetType();
                ar & type;
                std::string data = p->Serialize();
                ar & data;
            }
        } else {
            size_t count;
            ar & count;
            promises.resize(count);
            for (size_t i = 0; i < count; i++) {
                communication::VoidPromiseType type;
                ar & type;
                std::string data;
                ar & data;
                promises[i] = communication::CreateVoidPromiseFromSerialized(type, data);
            }
        }
    }
};

// ============================================================
// SyncOp
// ============================================================
//
// System operator that synchronizes one or more devices.
// - Zero inputs, zero outputs.
// - InferOperatorAssignInfo() assigns one stream key per device.
// - Each device gets a SyncKernel that synchronizes that device's
//   stream and fulfills the corresponding VoidPromise.

class SyncOp : public Operator {
public:
    SyncOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    size_t InferOutputSize() const override { return 0; }

    void InferOutputMetaInfo() const override {}

    void InferOperatorAssignInfo() override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    std::string GetDescribeString() const override;

    // Pure control-flow: no compute, no data moved.
    OperatorCost GetOperatorCost() const override { return OperatorCost{}; }
};

}  // namespace core
}  // namespace dtorch
