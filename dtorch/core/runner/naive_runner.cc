/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "naive_runner.h"

#include <cstdint>
#include <memory>

#include <torch/torch.h>

#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/filesystem.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/global_id_manager.h"
#include "dtorch/core/kernel/kernel_factory.h"
#include "dtorch/core/operators/operator_param.h"
#include "dtorch/core/type.h"
#include "dtorch/external/boost/boost_interprocess.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

NaiveRunner::NaiveRunner(const GraphOption& graphOption, const RunnerSupportedDevices& supportedDevices,
                         const communication::TensorStoreConfig& storeConfig)
    : mSupportedDevices(supportedDevices),
      mOperandToBlobs(),
      mThreadGroupManager(),
      mStreamManager(),
      mStoreConfig(storeConfig) {
    DDebugAssert(graphOption.graphId.has_value());
    DDebugAssert(graphOption.graphId != GraphIdManager::kNoValue);
    mThreadGroupManager = std::make_shared<communication::ThreadGroupManager>(
        graphOption.graphId.value(), core::GlobalOption::GetSingleton().GetDTensorInSameDevice());
}

void NaiveRunner::Execute(const std::vector<std::shared_ptr<Operator>>& ops,
                          const std::vector<const Operand*>& noHoldOperands) {
    // Assert here to avoid infinite loop
    DDebugAssert(ops.size() + noHoldOperands.size() > 0);
    std::vector<std::unique_ptr<Kernel>> kernels;

    for (auto op : ops) {
        auto tmpKernels = CreateKernelForOperator(op);
        for (auto& it : tmpKernels) {
            kernels.push_back(std::move(it));
        }
    }

    for (auto operand : noHoldOperands) {
        DeleteOperand(operand);
    }

    for (auto& kernel : kernels) {
        kernel->GetStream().LaunchKernel(std::move(kernel));
    }
}

std::vector<std::unique_ptr<Kernel>> NaiveRunner::CreateKernelForOperator(const std::shared_ptr<Operator>& op) {
    std::vector<std::unique_ptr<Kernel>> result;

    const OperatorAssignInfo& assignInfo = op->GetOperatorAssignInfo();
    size_t numKernelForThisOp = assignInfo.NumKernelForThisOp();
    DDebugAssert(numKernelForThisOp > 0);

    std::shared_ptr<communication::TensorStoreCreateInfo> tensorStoreCreateInfo = nullptr;
    if (numKernelForThisOp > 1) {
        std::stringstream ss;
        ss << "TensorStoreForKernel_" << op->GetUniqueId();
        const std::string shmFileName = external::boost::ManagedSharedMemory::GetShmFileNameWithPrefix(ss.str());
        tensorStoreCreateInfo =
            std::make_shared<communication::TensorStoreCreateInfo>(mStoreConfig, shmFileName, numKernelForThisOp);
    }

    // inputs
    std::vector<AllDeviceBlobs> inputs;
    for (auto operand : op->GetInputOperands()) {
        auto it = mOperandToBlobs.find(operand.get());
        DAlwaysAssert(it != mOperandToBlobs.end());
        inputs.push_back(it->second);
    }

    // outputs
    std::vector<AllDeviceBlobs> outputs;
    for (auto it : op->GetOutputOperands()) {
        Operand* operand = it.get();
        DAlwaysAssert(!mOperandToBlobs.count(operand)) mOperandToBlobs[operand] = std::unordered_map<int64_t, Blob>();
        for (auto globalDeviceId : operand->GetDeviceMesh().GetDeviceIdSet()) {
            DAlwaysAssert(!mOperandToBlobs[operand].count(globalDeviceId));
            NewBlob(operand, globalDeviceId);
        }
        outputs.push_back(mOperandToBlobs[operand]);
    }

    auto supportedStreamKeys = mSupportedDevices.GetSupported(assignInfo.GetStreamKeySet());

    for (const auto& streamKey : supportedStreamKeys) {
        Device globalDevice = streamKey.GetDevice();
        Device localDevice = mSupportedDevices.GlobalToLocal(globalDevice);

        // stream
        KernelStream* stream = mStreamManager.GetStream(globalDevice, localDevice, streamKey.streamType, true);

        // kernel
        KernelCreateCtx ctx(op, inputs, outputs, stream, globalDevice, localDevice, numKernelForThisOp,
                            mThreadGroupManager.get());
        std::unique_ptr<Kernel> kernel = KernelFactory::GetSingleton().NewKernel(ctx);
        if (tensorStoreCreateInfo) {
            kernel->SetTensorStoreCreateInfo(tensorStoreCreateInfo);
        }
        result.push_back(std::move(kernel));
    }

    return result;
}

void NaiveRunner::DeleteOperand(const Operand* operand) {
    DAlwaysAssert(mOperandToBlobs.count(operand));
    mOperandToBlobs.erase(operand);
}

Blob& NaiveRunner::NewBlob(const Operand* operand, int64_t globalDeviceId) {
    DDebugAssert(mOperandToBlobs.count(operand));  // Caller ensures outer map entry exists
    DAlwaysAssert(!mOperandToBlobs[operand].count(globalDeviceId));
    Blob newBlob;
    mOperandToBlobs[operand][globalDeviceId] = newBlob;
    return mOperandToBlobs[operand][globalDeviceId];
}

}  // namespace core
}  // namespace dtorch
