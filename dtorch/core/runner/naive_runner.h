/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "dtorch/core/blob.h"
#include "dtorch/core/communication/thread_group/thread_group_manager.h"
#include "dtorch/core/kernel_stream/kernel_stream_manager.h"
#include "dtorch/core/operators/operator.h"
#include "dtorch/core/runner/node_runner_base.h"
#include "dtorch/core/runner/runner_supported_devices.h"

namespace dtorch {
namespace core {

// load_state_dict 交给更上层去解决，NaiveRunner 这一层只处理 Tensor 及其切片。
// 获取 tensor 的值，需要上层增加通信 operator，将数据聚合到 master 节点。
class NaiveRunner {
public:
    NaiveRunner(const GraphOption& graphOption, const RunnerSupportedDevices& supportedDevices,
                const communication::TensorStoreConfig& storeConfig);

    ~NaiveRunner() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(NaiveRunner);

    DTORCH_FORCEINLINE KernelStreamManager& GetStreamManager() noexcept { return mStreamManager; }

    void Execute(const std::vector<std::shared_ptr<Operator>>& ops, const std::vector<const Operand*>& noHoldOperands);

    std::vector<std::unique_ptr<Kernel>> CreateKernelForOperator(const std::shared_ptr<Operator>& op);

    void DeleteOperand(const Operand* operand);

private:
    Blob& NewBlob(const Operand* operand, int64_t globalDeviceId);

private:
    RunnerSupportedDevices mSupportedDevices;
    std::unordered_map<const Operand*, AllDeviceBlobs> mOperandToBlobs;
    std::shared_ptr<communication::ThreadGroupManager> mThreadGroupManager;
    KernelStreamManager mStreamManager;
    const communication::TensorStoreConfig mStoreConfig;
};

}  // namespace core
}  // namespace dtorch
