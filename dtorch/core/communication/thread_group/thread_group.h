/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/common/utilities.h"
#include "dtorch/core/type.h"
#include "dtorch/external/device/device_stream.h"
#include "dtorch/external/torch/torch_util.h"

using dtorch::external::device::DeviceStream;

namespace dtorch {
namespace core {
namespace communication {

enum class ReduceOpType {
    kSum = 0,
    kAvg,
    kProduce,
    kMin,
    kMax,
    kCount,
};

class ThreadGroup {
public:
    ThreadGroup(const std::string& initPath, DeviceKind deviceKind, const std::vector<int64_t>& allGlobalDeviceId,
                int rank, int size, bool sameDevice);

    ~ThreadGroup();

    DTORCH_DISABLE_COPY_AND_MOVE(ThreadGroup);

    int GetRank() const;

    int GetWorldSize() const;

    int64_t GetGlobalDeviceId() const;

    void SetStream(DeviceStream& stream);

    void Barrier();

    // send, recv, broadcast, all_reduce, reduce, all_gather, gather, scatter, reduce_scatter

    // P -> R
    torch::Tensor AllReduce(torch::Tensor& input, ReduceOpType reduceOpType = ReduceOpType::kSum);

    // SO -> R
    // torch primitive not support uneven sized tensors
    torch::Tensor AllGatherIntoTensor(torch::Tensor& input, const Operand& inputOperand);
    torch::Tensor EqualShapeAllGatherIntoTensor(torch::Tensor& input);

    // S -> R
    // torch primitive support uneven sized tensors
    torch::Tensor AllGather(torch::Tensor& input, const Operand& inputOperand, int64_t shardIndex);

    std::vector<torch::Tensor> AllGatherIntoVec(torch::Tensor& input, const Operand& inputOperand);

    // P -> S0
    // torch primitive not support uneven sized tensors
    torch::Tensor ReduceScatterTensor(torch::Tensor& input, ReduceOpType reduceOpType = ReduceOpType::kSum);

    // S -> S
    // torch primitive support uneven sized tensors
    torch::Tensor AllToAll(torch::Tensor& input, size_t srcDim, size_t destDim, const Operand& inputOperand);

    // R -> S; \p subSplitCoordinates < 0 means no sub-split (uniform shard along \p shardIdx).
    torch::Tensor ReplicateToShard(torch::Tensor& input, int64_t shardIdx, int64_t subSplitCoordinates = -1);

private:
    struct Impl;
    std::shared_ptr<Impl> mImplPtr;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
