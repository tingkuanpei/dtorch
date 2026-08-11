/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "distributed_scather_and_gather.h"

#include <cstdint>
#include <memory>

#include <ATen/ops/zeros_like.h>
#include <torch/torch.h>

#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

DistributedScather::DistributedScather(const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq,
                                       const torch::Tensor& torchTensor)
    : mDeviceMesh(deviceMesh), mPlacementSeq(placementSeq), mResult() {
    DDebugAssert(deviceMesh.NumAxis() == mPlacementSeq.Size());
    DistributedSpec::MergeSameShardDim(mDeviceMesh, mPlacementSeq);
    ScatherImp(torchTensor, mDeviceMesh.GetMesh().ToTrochTensor(), 0);
}

torch::Tensor DistributedScather::Get(int64_t globalDeviceId) {
    auto it = mResult.find(globalDeviceId);
    DDebugAssert(it != mResult.end());
    torch::Tensor& result = *(it->second);
    return result;
}

void DistributedScather::ScatherImp(torch::Tensor torchTensor, const torch::Tensor& mesh, size_t index) {
    if (index == mPlacementSeq.Size()) {
        DDebugAssert(mesh.numel() == 1);
        int64_t globalDeviceId = external::torch::TorchUtil::ToInt64Vec(mesh)[0];
        DDebugAssert(mResult.count(globalDeviceId) == 0);
        mResult[globalDeviceId] = std::make_shared<torch::Tensor>(torchTensor);
        return;
    }

    DDebugAssert(mesh.dim() > 0);
    int64_t numOfTensor = mesh.size(0);
    std::vector<torch::Tensor> tensors;

    if (mPlacementSeq[index].IsReplicate()) {
        tensors = std::vector<torch::Tensor>(numOfTensor, torchTensor);
    } else if (mPlacementSeq[index].IsShard()) {
        int64_t shardIndex = mPlacementSeq[index].GetShardIndex();
        int64_t subSplitCoordinates = mPlacementSeq[index].GetSubSplitCoordinates();
        if (subSplitCoordinates == -1) {
            std::vector<int64_t> splitSize =
                Placement::GetShardSizeForAllRank(torchTensor.size(shardIndex), numOfTensor);
            tensors = torch::split(torchTensor, splitSize, shardIndex);
        } else {
            int64_t sizeInShardDim = torchTensor.size(shardIndex);
            DDebugAssert(subSplitCoordinates < sizeInShardDim);

            std::vector<int64_t> towSplitSize = {subSplitCoordinates + 1, (sizeInShardDim - subSplitCoordinates - 1)};
            std::vector<torch::Tensor> towTensors = torch::split(torchTensor, towSplitSize, shardIndex);
            DDebugAssert(towTensors.size() == 2);

            std::vector<int64_t> frontSplitSize =
                Placement::GetShardSizeForAllRank(subSplitCoordinates + 1, numOfTensor);
            std::vector<torch::Tensor> frontTensors = torch::split(towTensors[0], frontSplitSize, shardIndex);

            std::vector<int64_t> endSplitSize =
                Placement::GetShardSizeForAllRank(sizeInShardDim - subSplitCoordinates - 1, numOfTensor);
            std::vector<torch::Tensor> endTensors = torch::split(towTensors[1], endSplitSize, shardIndex);

            DDebugAssert(frontTensors.size() == endTensors.size());
            for (size_t i = 0; i < frontTensors.size(); i++) {
                tensors.push_back(torch::concat({frontTensors[i], endTensors[i]}, shardIndex));
            }
        }
    } else {
        tensors = std::vector<torch::Tensor>(numOfTensor, torch::zeros_like(torchTensor));
        tensors[0] = torchTensor;
    }

    DDebugAssert(static_cast<int64_t>(tensors.size()) == numOfTensor);
    for (int64_t i = 0; i < numOfTensor; i++) {
        ScatherImp(tensors[i], mesh[i], index + 1);
    }
}

DistributeGather::DistributeGather(const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq,
                                   const std::unordered_map<int64_t, std::shared_ptr<torch::Tensor>>& torchMap)
    : mDeviceMesh(deviceMesh), mPlacementSeq(placementSeq), mTorchMap(torchMap) {
    DDebugAssert(torchMap.size() == static_cast<size_t>(mDeviceMesh.GetMesh().Count()));
    DDebugAssert(torchMap.size() > 0);

    // Check all torch tensor in same device
    Device expectDevice = external::torch::TorchUtil::GetDevice(*(torchMap.begin()->second));
    for (auto& it : torchMap) {
        DDebugAssert(expectDevice == external::torch::TorchUtil::GetDevice(*it.second));
    }
}

torch::Tensor DistributeGather::Get() { return GatherImp(0, mDeviceMesh.GetMesh().ToTrochTensor()); }

torch::Tensor DistributeGather::GatherImp(size_t index, const torch::Tensor& mesh) {
    if (index == mPlacementSeq.Size()) {
        DDebugAssert(mesh.numel() == 1);
        int64_t deviceId = external::torch::TorchUtil::ToInt64Vec(mesh)[0];
        DDebugAssert(mTorchMap.count(deviceId) == 1);
        return *(mTorchMap.at(deviceId));
    }

    DDebugAssert(mesh.dim() > 0);
    int64_t numOfTensor = mesh.size(0);
    std::vector<torch::Tensor> gatherTensor;

    if (mPlacementSeq[index].IsReplicate()) {
        return GatherImp(index + 1, mesh[0]);
    } else if (mPlacementSeq[index].IsShard()) {
        for (int64_t i = 0; i < numOfTensor; i++) {
            gatherTensor.push_back(GatherImp(index + 1, mesh[i]));
        }

        int64_t subSplitCoordinates = mPlacementSeq[index].GetSubSplitCoordinates();
        if (subSplitCoordinates == -1) {
            int64_t shardIndex = mPlacementSeq[index].GetShardIndex();
            return torch::concat(gatherTensor, shardIndex);
        } else {
            return GatherShardWithSubSplit(gatherTensor, mDeviceMesh, mPlacementSeq, index);
        }
    } else {
        torch::Tensor result;
        for (int64_t i = 0; i < numOfTensor; i++) {
            gatherTensor.push_back(GatherImp(index + 1, mesh[i]));
        }
        result = gatherTensor[0];
        for (int64_t i = 1; i < numOfTensor; i++) {
            result = result + gatherTensor[i];
        }
        return result;
    }
}

torch::Tensor DistributeGather::GatherShardWithSubSplit(const std::vector<torch::Tensor>& tensors,
                                                        const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq,
                                                        size_t index) {
    int64_t shardIndex = placementSeq[index].GetShardIndex();
    int64_t subSplitCoordinates = placementSeq[index].GetSubSplitCoordinates();
    DDebugAssert(subSplitCoordinates > 0);

    // TODO: dirty code!!! when encountering bugs, please rewrite this part of the code.
    size_t shardSize = tensors.size();
    DDebugAssert(tensors.size() > 0);
    if (subSplitCoordinates > tensors[0].size(shardIndex)) {
        std::vector<DistributedSpec::ShardInfo> shardInfo = DistributedSpec::GetShardInfo(deviceMesh, placementSeq);
        for (const auto& it : shardInfo) {
            if (static_cast<int64_t>(it.shardDim) == shardIndex) {
                shardSize = it.shardSize;
                break;
            }
        }
    }

    std::vector<int64_t> splitSize = Placement::GetShardSizeForAllRank(subSplitCoordinates + 1, shardSize);
    splitSize.resize(tensors.size());
    DDebugAssert(splitSize.size() == tensors.size());

    std::vector<torch::Tensor> tensorsFront;
    std::vector<torch::Tensor> tensorsEnd;
    for (size_t i = 0; i < tensors.size(); i++) {
        int64_t tensorSizeInDim = tensors[i].size(shardIndex);
        DAlwaysAssert(splitSize[i] < tensorSizeInDim);
        std::vector<int64_t> subSplitSize = {splitSize[i], tensorSizeInDim - splitSize[i]};
        auto tmpTensors = torch::split(tensors[i], subSplitSize, shardIndex);
        tensorsFront.push_back(tmpTensors[0]);
        tensorsEnd.push_back(tmpTensors[1]);
    }

    return torch::concat({torch::concat(tensorsFront, shardIndex), torch::concat(tensorsEnd, shardIndex)}, shardIndex);
}

}  // namespace core
}  // namespace dtorch
