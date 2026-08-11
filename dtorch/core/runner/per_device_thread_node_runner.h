/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/runner/naive_runner.h"
#include "dtorch/core/runner/node_runner_base.h"
#include "dtorch/core/runner/runner_supported_devices.h"

namespace dtorch {
namespace core {

class PerDeviceThreadNodeRunner : public NodeRunnerBase {
public:
    PerDeviceThreadNodeRunner(const GraphOption& graphOption, const RunnerSupportedDevices& supportedDevices)
        : mNaiveRunner(graphOption, supportedDevices,
                       communication::TensorStoreConfig(communication::TensorStoreType::kMemory)) {}

    DTORCH_FORCEINLINE void Execute(const std::vector<std::shared_ptr<Operator>>& ops,
                                    const std::vector<const Operand*>& noHoldOperands) override {
        mNaiveRunner.Execute(ops, noHoldOperands);
    }

private:
    NaiveRunner mNaiveRunner;
};

}  // namespace core
}  // namespace dtorch
