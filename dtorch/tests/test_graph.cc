/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <torch/torch.h>
#include <torch/types.h>

#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/dtorch.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/external/torch/torch_util.h"
#include "test.h"

using namespace dtorch::api::cpp;
using namespace dtorch::core;
using namespace dtorch::external::torch;

void DynamicGraphAddImp(Device device, bool perDevicePerProcess) {
    torch::Tensor torchTensorA = torch::rand({4, 3}).to(TorchUtil::ToDevice(device));
    torch::Tensor torchTensorB = torch::rand({4, 3}).to(TorchUtil::ToDevice(device));
    torch::Tensor torchTensorC = torchTensorA.add(torchTensorB);

    GraphOption option;
    option.perDevicePerProcess = perDevicePerProcess;
    ug::Graph graph(option);

    ug::Tensor tensorA(graph, torchTensorA);
    ug::Tensor tensorB(graph, torchTensorB);
    ug::Tensor tensorC = ug::functional::_Add(tensorA, tensorB);
    torch::Tensor fromTensorC = tensorC.GetTorchTensor();

    EXPECT_TRUE(torch::allclose(torchTensorC, fromTensorC));
}

void DistributedDynamicGraphAddImp(Device device, bool perDevicePerProcess) {
    torch::Tensor torchTensorA = torch::rand({4, 3}).to(TorchUtil::ToDevice(device));
    torch::Tensor torchTensorB = torch::rand({4, 3}).to(TorchUtil::ToDevice(device));
    torch::Tensor torchTensorC = torchTensorA.add(torchTensorB);

    GraphOption option;
    option.perDevicePerProcess = perDevicePerProcess;
    ug::Graph graph(option);
    ug::DeviceMesh deviceMesh(device.deviceKind, {0, 1, 2, 3});
    if (!graph.Satisfy(deviceMesh)) {
        return;
    }
    ug::PlacementSeq placementSeq = {ug::Shard(0)};
    ug::Tensor tensorA(graph, torchTensorA, deviceMesh, placementSeq);
    ug::Tensor tensorB(graph, torchTensorB, deviceMesh, placementSeq);
    ug::Tensor tensorC = ug::functional::_Add(tensorA, tensorB);
    torch::Tensor fromTensorC = tensorC.GetTorchTensor();

    EXPECT_TRUE(torch::allclose(torchTensorC, fromTensorC));
    EXPECT_TRUE(placementSeq == tensorC.GetPlacementSeq());
    EXPECT_TRUE(deviceMesh == tensorC.GetDeviceMesh());
}

TEST(GraphTest, DynamicGraphAdd) {
    DynamicGraphAddImp(DeviceKind::kCpu, true);
    DynamicGraphAddImp(DeviceKind::kCpu, false);

    if (Device::IsAvailable(DeviceKind::kGpu)) {
        DynamicGraphAddImp(DeviceKind::kGpu, true);
        DynamicGraphAddImp(DeviceKind::kGpu, false);
    }
}

TEST(GraphTest, DistributedDynamicGraphAdd) {
    DistributedDynamicGraphAddImp(DeviceKind::kCpu, true);
    DistributedDynamicGraphAddImp(DeviceKind::kCpu, false);

    if (Device::IsAvailable(DeviceKind::kGpu)) {
        DistributedDynamicGraphAddImp(DeviceKind::kGpu, true);
        DistributedDynamicGraphAddImp(DeviceKind::kGpu, false);
    }
}
