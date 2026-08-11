/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "remote_runner.h"

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/operators/operator_factory.h"
#include "dtorch/core/operators/operator_serialization_pack.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/cuda/nvtx_profiler.h"
#include "dtorch/external/zmq/remote_runner_publisher.h"

namespace dtorch {
namespace core {

RemoteRunner::RemoteRunner(const std::string& pushAddress, const std::string& publisherAddress,
                           const GraphOption& graphOption, const RunnerSupportedDevices& supportedDevices,
                           const communication::TensorStoreConfig& storeConfig)
    : mNaiveRunner(graphOption, supportedDevices, storeConfig),
      mOperandMap(),
      mSupportedDevices(supportedDevices),
      mPusher(pushAddress),
      mSubscriber(publisherAddress),
      mAsyncThread(),
      mGetDestroySignal(false) {
    mAsyncThread = std::thread(&RemoteRunner::AsyncMain, this);
}

RemoteRunner::~RemoteRunner() {
    mGetDestroySignal = true;
    if (mAsyncThread.joinable()) {
        mAsyncThread.join();
    }
}

void RemoteRunner::AsyncMain() {
    UNvtxNameOsThread("DTorchRemoteRunnerThread");

    try {
        mPusher.NotifyDevicesReady(mSupportedDevices.AllDeviceList());

        while (true) {
            if (mGetDestroySignal) {
                break;
            }

            bool hasNewMessage = mSubscriber.Get();
            const std::string& messageType = mSubscriber.GetMessageType();

            if (hasNewMessage) {
                if (messageType == external::zmq::RemoteRunnerPublisher::kDestroyStr) {
                    mGetDestroySignal = true;
                } else if (messageType == external::zmq::RemoteRunnerPublisher::kExecuteStr) {
                    ProcessSubscriberExecuteMessage(mSubscriber.GetSerializedData());
                }
            }
        }
    } catch (std::exception& e) {
        DLogFatal() << "RemoteRunner::AsyncMain got exception: " << e.what();
    }
}

void RemoteRunner::ProcessSubscriberExecuteMessage(const std::string& serializedData) {
    std::vector<OperatorSerializationPack> opPacks;
    std::vector<uintptr_t> noHoldOperandPtrs;
    std::stringstream ss(serializedData, std::ios::in | std::ios::binary);
    external::boost::BinaryIArchive bia(ss);
    bia >> opPacks;
    bia >> noHoldOperandPtrs;
    DDebugAssert(opPacks.size() + noHoldOperandPtrs.size() > 0);

    ExecuteSerialization(opPacks, noHoldOperandPtrs);
}

void RemoteRunner::ExecuteSerialization(const std::vector<OperatorSerializationPack>& opPacks,
                                        const std::vector<uintptr_t>& uintNoHoldOperands) {
    std::vector<std::shared_ptr<Operator>> ops;
    for (const auto& opPack : opPacks) {
        OperandArray inputOperands;
        for (auto it : opPack.uintInputOperands) {
            DAlwaysAssert(mOperandMap.count(it) > 0);
            inputOperands.push_back(mOperandMap[it]);
        }

        try {
            std::shared_ptr<Operator> op =
                OperatorFactory::GetSingleton().NewOperatorOrThrow(opPack.opParam, inputOperands, opPack.uniqueId);
            OperandArray outputOperands = op->GetOutputOperands();
            DAlwaysAssert(outputOperands.size() == opPack.uintOutputOperands.size());
            for (size_t i = 0; i < outputOperands.size(); i++) {
                mOperandMap[opPack.uintOutputOperands[i]] = outputOperands[i];
            }
            ops.push_back(op);
        } catch (std::exception& e) {
            DLogFatal() << e.what();
        }
    }

    std::vector<const Operand*> noHoldOperands;
    for (auto it : uintNoHoldOperands) {
        DAlwaysAssert(mOperandMap.count(it) > 0);
        noHoldOperands.push_back(mOperandMap[it].get());
        mOperandMap.erase(it);
    }

    Execute(ops, noHoldOperands);
}

void RemoteRunner::Execute(const std::vector<std::shared_ptr<Operator>>& ops,
                           const std::vector<const Operand*>& noHoldOperands) {
    mNaiveRunner.Execute(ops, noHoldOperands);
}

}  // namespace core
}  // namespace dtorch
