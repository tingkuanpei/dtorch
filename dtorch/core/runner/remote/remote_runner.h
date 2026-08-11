/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <unordered_map>

#include "dtorch/common/utilities.h"
#include "dtorch/core/communication/tensor_store/tensor_store.h"
#include "dtorch/core/runner/naive_runner.h"
#include "dtorch/core/runner/runner_supported_devices.h"
#include "dtorch/external/zmq/remote_runner_pusher.h"
#include "dtorch/external/zmq/remote_runner_subscriber.h"

namespace dtorch {
namespace core {

class OperatorSerializationPack;

class RemoteRunner {
public:
    RemoteRunner(const std::string& pushAddress, const std::string& publisherAddress, const GraphOption& graphOption,
                 const RunnerSupportedDevices& supportedDevices, const communication::TensorStoreConfig& storeConfig);

    ~RemoteRunner();

    DTORCH_DISABLE_COPY_AND_MOVE(RemoteRunner);

private:
    void AsyncMain();

    void ProcessSubscriberExecuteMessage(const std::string& serializedData);

    void ExecuteSerialization(const std::vector<OperatorSerializationPack>& opPacks,
                              const std::vector<uintptr_t>& uintNoHoldOperands);

    void Execute(const std::vector<std::shared_ptr<Operator>>& ops, const std::vector<const Operand*>& noHoldOperands);

    NaiveRunner mNaiveRunner;
    std::unordered_map<uintptr_t, std::shared_ptr<Operand>> mOperandMap;
    RunnerSupportedDevices mSupportedDevices;
    external::zmq::RemoteRunnerPusher mPusher;
    external::zmq::RemoteRunnerSubscriber mSubscriber;
    std::thread mAsyncThread;
    std::atomic_bool mGetDestroySignal;
};

}  // namespace core
}  // namespace dtorch
