/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "thread_group.h"

#include <memory>

#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/torch.h>

#include "dtorch/common/config.h"
#if DTORCH_WITH_CUDA
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/filesystem.h"
#include "dtorch/common/type_cast.h"
#include "dtorch/core/operand.h"
#include "dtorch/external/torch/torch_util.h"
#include "thread_group_same_device_backend.h"
#if DTORCH_WITH_CUDA
#include "dtorch/external/cuda/cuda_device.h"
#endif

namespace dtorch {
namespace core {
namespace communication {

struct ThreadGroup::Impl {
    Impl(DeviceKind deviceKind, const std::vector<int64_t>& allGlobalDeviceId, int rank, int size, bool sameDevice)
        : deviceKind(deviceKind),
          rank(rank),
          worldSize(size),
          sameDevice(sameDevice),
          cudaDevice(0),
          allGlobalDeviceId(allGlobalDeviceId),
          backend() {
        DDebugAssert(allGlobalDeviceId.size() == static_cast<size_t>(size));
#if DTORCH_WITH_CUDA
        if (deviceKind == DeviceKind::kGpu) {
            cudaDevice = external::cuda::CudaDevice::GetDevice();
        }
#endif
    }

    DeviceKind deviceKind;
    int rank;
    int worldSize;
    bool sameDevice;
    int cudaDevice;
    std::vector<int64_t> allGlobalDeviceId;
    std::unique_ptr<c10d::Backend> backend;
};

ThreadGroup::ThreadGroup(const std::string& initPath, DeviceKind deviceKind,
                         const std::vector<int64_t>& allGlobalDeviceId, int rank, int size, bool sameDevice)
    : mImplPtr(std::make_shared<Impl>(deviceKind, allGlobalDeviceId, rank, size, sameDevice)) {
    if (sameDevice) {
        mImplPtr->backend =
            std::unique_ptr<c10d::Backend>(new ThreadGroupSameDeviceBackend(initPath, deviceKind, rank, size));
    } else {
        DDebugAssert(!initPath.empty());
        // c10d::FileStore: for single machine multi process.
        // c10d::TCPStore: for multi machine multi process.
        std::string initPathInTmpDir = GetTempDirectoryPath() + "/" + initPath;
        auto store = c10::make_intrusive<::c10d::FileStore>(initPathInTmpDir, size);
        IgnoreUnused(store);

        switch (deviceKind) {
#if DTORCH_WITH_CUDA
            case DeviceKind::kGpu: {
                c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts =
                    c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
                opts->timeout = std::chrono::seconds(GlobalOption::GetSingleton().GetCommTimeoutSecond());
                mImplPtr->backend =
                    std::unique_ptr<c10d::Backend>(new c10d::ProcessGroupNCCL(store, rank, size, std::move(opts)));
            } break;
#endif
            default:
                DUnimplemented();
                break;
        }
    }
}

ThreadGroup::~ThreadGroup() {
#if DTORCH_WITH_CUDA
    DDebugAssert(mImplPtr->backend != nullptr);
    if (mImplPtr->deviceKind == DeviceKind::kGpu && !mImplPtr->sameDevice) {
        c10d::ProcessGroupNCCL* pgNccl = dynamic_cast<c10d::ProcessGroupNCCL*>(mImplPtr->backend.get());
        DDebugAssert(pgNccl != nullptr);
        pgNccl->shutdown();
    }
#endif
}

int ThreadGroup::GetRank() const { return mImplPtr->rank; }

int ThreadGroup::GetWorldSize() const { return mImplPtr->worldSize; }

int64_t ThreadGroup::GetGlobalDeviceId() const { return mImplPtr->allGlobalDeviceId[GetRank()]; }

void ThreadGroup::SetStream(DeviceStream& stream) {
    if (mImplPtr->sameDevice) {
        ThreadGroupSameDeviceBackend* threadGroupSameDeviceBackend =
            DerivedCast<ThreadGroupSameDeviceBackend, c10d::Backend>(mImplPtr->backend.get());
        DDebugAssert(threadGroupSameDeviceBackend != nullptr);
        threadGroupSameDeviceBackend->SetStream(stream);
    }
}

void ThreadGroup::Barrier() {
    DDebugAssert(mImplPtr->backend != nullptr);
    c10d::BarrierOptions option;
    if (mImplPtr->deviceKind == DeviceKind::kGpu) {
        option.device_ids = {mImplPtr->cudaDevice};
    }
    mImplPtr->backend->barrier(option)->wait();
}

torch::Tensor ThreadGroup::AllReduce(torch::Tensor& input, ReduceOpType reduceOpType) {
    auto timeout = std::chrono::seconds(GlobalOption::GetSingleton().GetCommTimeoutSecond());
    c10d::AllreduceOptions opts;
    DDebugAssert(reduceOpType == ReduceOpType::kSum);
    opts.reduceOp = c10d::ReduceOp::SUM;
    opts.timeout = timeout;

    // Clone the input because NCCL's allreduce operates in-place, which would
    // mutate the original tensor if the caller still needs it (e.g., for backward pass).
    std::vector<at::Tensor> inputs = {input.clone()};
    auto work = mImplPtr->backend->allreduce(inputs, opts);
    work->synchronize();

    return inputs[0];
}

torch::Tensor ThreadGroup::AllGatherIntoTensor(torch::Tensor& input, const Operand& inputOperand) {
    if (inputOperand.IsLocalShapeEventSplit()) {
        return ThreadGroup::EqualShapeAllGatherIntoTensor(input);
    } else {
        return ThreadGroup::AllGather(input, inputOperand, 0);
    }
}

torch::Tensor ThreadGroup::EqualShapeAllGatherIntoTensor(torch::Tensor& input) {
    Shape outputShape = external::torch::TorchUtil::GetShape(input);
    DDebugAssert(outputShape.NumAxis() > 0);
    outputShape[0] = outputShape[0] * GetWorldSize();
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty(outputShape.Vec(), options);

    c10d::AllgatherOptions opts;
    auto timeout = std::chrono::seconds(GlobalOption::GetSingleton().GetCommTimeoutSecond());
    opts.timeout = timeout;
    torch::Tensor inputContigours = input.contiguous();
    auto work = mImplPtr->backend->_allgather_base(output, inputContigours, opts);
    work->synchronize();

    return output;
}

torch::Tensor ThreadGroup::AllGather(torch::Tensor& input, const Operand& inputOperand, int64_t shardIndex) {
    std::vector<torch::Tensor> result = AllGatherIntoVec(input, inputOperand);
    return torch::concat(result, shardIndex);
}

std::vector<torch::Tensor> ThreadGroup::AllGatherIntoVec(torch::Tensor& input, const Operand& inputOperand) {
    std::vector<at::Tensor> inputTensors = {input.contiguous()};
    std::vector<std::vector<at::Tensor>> outputTensors;
    outputTensors.push_back(std::vector<at::Tensor>());
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());

    for (auto id : mImplPtr->allGlobalDeviceId) {
        outputTensors[0].push_back(torch::empty(inputOperand.GetLocalShape(id).Vec(), options));
    }

    c10d::AllgatherOptions opts;
    auto timeout = std::chrono::seconds(GlobalOption::GetSingleton().GetCommTimeoutSecond());
    opts.timeout = timeout;

    auto work = mImplPtr->backend->allgather(outputTensors, inputTensors, opts);
    work->synchronize();
    return outputTensors[0];
}

torch::Tensor ThreadGroup::ReduceScatterTensor(torch::Tensor& input, ReduceOpType reduceOpType) {
    Shape outputShape = external::torch::TorchUtil::GetShape(input);
    DDebugAssert(outputShape.NumAxis() > 0);
    DDebugAssert(outputShape[0] % GetWorldSize() == 0);
    outputShape[0] = outputShape[0] / GetWorldSize();
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty(outputShape.Vec(), options);

    c10d::ReduceScatterOptions opts;
    DDebugAssert(reduceOpType == ReduceOpType::kSum);
    opts.reduceOp = c10d::ReduceOp::SUM;
    auto timeout = std::chrono::seconds(GlobalOption::GetSingleton().GetCommTimeoutSecond());
    opts.timeout = timeout;
    auto work = mImplPtr->backend->_reduce_scatter_base(output, input, opts);
    work->synchronize();

    return output;
}

torch::Tensor ThreadGroup::AllToAll(torch::Tensor& input, size_t srcDim, size_t destDim, const Operand& inputOperand) {
    Shape inputShape = external::torch::TorchUtil::GetShape(input);
    DDebugAssert(inputShape.NumAxis() > srcDim);
    DDebugAssert(inputShape.NumAxis() > destDim);

    std::vector<int64_t> splitSize = Placement::GetShardSizeForAllRank(input.size(destDim), GetWorldSize());
    auto inputVec = torch::split(input, splitSize, destDim);
    for (auto& it : inputVec) {
        it = it.contiguous();
    }
    DDebugAssert(inputVec.size() == static_cast<size_t>(GetWorldSize()));

    c10d::AllToAllOptions opts;
    auto timeout = std::chrono::seconds(GlobalOption::GetSingleton().GetCommTimeoutSecond());
    opts.timeout = timeout;

    std::vector<torch::Tensor> outputVec;
    size_t worldSize = GetWorldSize();
    for (size_t i = 0; i < worldSize; i++) {
        Shape shape = inputOperand.GetLocalShape(mImplPtr->allGlobalDeviceId[i]);
        std::vector<int64_t> splitSize = Placement::GetShardSizeForAllRank(shape[destDim], worldSize);
        shape[destDim] = splitSize[GetRank()];

        auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
        outputVec.push_back(torch::empty(shape.Vec(), options));
    }
    auto work = mImplPtr->backend->alltoall(outputVec, inputVec, opts);
    work->synchronize();

    return torch::concat(outputVec, srcDim);
}

torch::Tensor ThreadGroup::ReplicateToShard(torch::Tensor& input, int64_t shardIdx, int64_t subSplitCoordinates) {
    Shape inputShape = external::torch::TorchUtil::GetShape(input);
    DDebugAssert(shardIdx >= 0);
    DDebugAssert(inputShape.NumAxis() > static_cast<size_t>(shardIdx));

    int64_t worldSize = static_cast<int64_t>(GetWorldSize());
    int64_t rank = static_cast<int64_t>(GetRank());
    int64_t sizeInShardDim = input.size(shardIdx);

    if (subSplitCoordinates < 0) {
        std::vector<int64_t> splitSize = Placement::GetShardSizeForAllRank(sizeInShardDim, worldSize);
        auto inputVec = torch::split(input, splitSize, shardIdx);
        DDebugAssert(inputVec.size() == static_cast<size_t>(worldSize));
        return inputVec[rank];
    } else {
        if (subSplitCoordinates <= 0 || subSplitCoordinates >= sizeInShardDim) {
            DLogFatal() << "Invalid subSplitCoordinates: " << subSplitCoordinates
                        << " and sizeInShardDim: " << sizeInShardDim;
        }
        std::vector<int64_t> towSplitSize = {subSplitCoordinates + 1, sizeInShardDim - subSplitCoordinates - 1};
        std::vector<torch::Tensor> towTensors = torch::split(input, towSplitSize, shardIdx);
        DDebugAssert(towTensors.size() == 2);

        std::vector<int64_t> frontSplitSize = Placement::GetShardSizeForAllRank(subSplitCoordinates + 1, worldSize);
        std::vector<torch::Tensor> frontTensors = torch::split(towTensors[0], frontSplitSize, shardIdx);

        std::vector<int64_t> endSplitSize =
            Placement::GetShardSizeForAllRank(sizeInShardDim - subSplitCoordinates - 1, worldSize);
        std::vector<torch::Tensor> endTensors = torch::split(towTensors[1], endSplitSize, shardIdx);

        DDebugAssert(frontTensors.size() == endTensors.size());
        DDebugAssert(static_cast<size_t>(rank) < frontTensors.size());
        return torch::concat({frontTensors[static_cast<size_t>(rank)], endTensors[static_cast<size_t>(rank)]},
                             shardIdx);
    }
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
