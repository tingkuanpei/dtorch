/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/runner/node_runner_base.h"
#include "dtorch/core/runner/remote/naive_remote_runner.h"
#include "dtorch/core/runner/runner_supported_devices.h"

namespace dtorch {
namespace core {

// class RemotePerDeviceThreadNodeRunner : public NodeRunnerBase {
// public:
//     RemotePerDeviceThreadNodeRunner(const GraphOption& graphOption, const RunnerSupportedDevices& supportedDevices,
//                                     const communication::TensorStoreConfig& storeConfig)
//         : mNaiveRemoteRunner(graphOption, supportedDevices, storeConfig) {}

//     DTORCH_FORCEINLINE void Execute(const std::vector<std::shared_ptr<Operator>>& ops,
//                                     const std::vector<const Operand*>& noHoldOperands) override {
//         mNaiveRemoteRunner.Execute(ops, noHoldOperands);
//     }

// private:
//     NaiveRemoteRunner mNaiveRemoteRunner;
// };

}  // namespace core
}  // namespace dtorch
