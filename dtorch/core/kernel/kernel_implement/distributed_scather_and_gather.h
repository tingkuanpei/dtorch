/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <unordered_map>

#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

class DistributedScather {
public:
    DistributedScather(const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq,
                       const torch::Tensor& torchTensor);

    torch::Tensor Get(int64_t globalDeviceId);

private:
    void ScatherImp(torch::Tensor torchTensor, const torch::Tensor& mesh, size_t index);

private:
    DeviceMesh mDeviceMesh;
    PlacementSeq mPlacementSeq;
    std::unordered_map<int64_t, std::shared_ptr<torch::Tensor>> mResult;
};

class DistributeGather {
public:
    DistributeGather(const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq,
                     const std::unordered_map<int64_t, std::shared_ptr<torch::Tensor>>& torchMap);

    torch::Tensor Get();

public:
    static torch::Tensor GatherShardWithSubSplit(const std::vector<torch::Tensor>& tensors,
                                                 const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq,
                                                 size_t index);

private:
    torch::Tensor GatherImp(size_t index, const torch::Tensor& mesh);

private:
    DeviceMesh mDeviceMesh;
    PlacementSeq mPlacementSeq;
    std::unordered_map<int64_t, std::shared_ptr<torch::Tensor>> mTorchMap;
};

}  // namespace core
}  // namespace dtorch
