/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <memory>

#include "dtorch/api/cpp/device.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/global_id_manager.h"
#include "dtorch/core/operators/operator_factory.h"
#include "dtorch/core/operators/operator_param.h"
#include "dtorch/core/operators/standard/create_op.h"
#include "dtorch/core/runner/remote/remote_runner_in_process.h"
#include "dtorch/external/zmq/remote_runner_publisher.h"
#include "dtorch/external/zmq/remote_runner_puller.h"
#include "dtorch/external/zmq/remote_runner_pusher.h"
#include "dtorch/external/zmq/zmq.h"
#include "test.h"

using namespace dtorch;
using namespace dtorch::core;
using namespace dtorch::api::cpp;
using namespace dtorch::api::cpp::distributed;

class RemoteRunnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure clean state: stop any leftover cluster from other tests
        if (Cluster::GetSingleton().IsCreate()) {
            if (Cluster::GetSingleton().GetNodeType() == NodeType::kMain) {
                MainNode::Stop();
            }
        }
    }

    void TearDown() override {
        if (Cluster::GetSingleton().IsCreate()) {
            if (Cluster::GetSingleton().GetNodeType() == NodeType::kMain) {
                MainNode::Stop();
            }
        }
    }
};

TEST_F(RemoteRunnerTest, RunnerSupportedDevicesTest) {
    DevicePair pair(Device(DeviceKind::kGpu, 0), Device(DeviceKind::kGpu, 0));
    RunnerSupportedDevices supportedDevices(pair);

    Device globalDevice(DeviceKind::kGpu, 0);
    Device localDevice = supportedDevices.GlobalToLocal(globalDevice);

    EXPECT_TRUE(globalDevice == localDevice);
}

TEST_F(RemoteRunnerTest, SimpleTest) {
    if (!Device::IsAvailable(DeviceKind::kGpu)) {
        return;
    }

    MainNode::SetMainNodeAddress("127.0.0.1:13000");
    ASSERT_TRUE(MainNode::Start());

    // 1. Create server and client
    DevicePair pair(Device(DeviceKind::kGpu, 0), Device(DeviceKind::kGpu, 0));
    RunnerSupportedDevices supportedDevices(pair);
    GraphOption option;
    const std::string publisherAddress = external::zmq::GetRandomZmqIpcAddress();
    const std::string pushPullAddress = external::zmq::GetRandomZmqIpcAddress();
    external::zmq::RemoteRunnerPublisher remoteRunnerPublisher(publisherAddress);
    external::zmq::RemoteRunnerPuller puller(pushPullAddress);
    RemoteRunnerInProcess remoteRunner(GraphOption::UpdateOptionFromEnvironment(option), supportedDevices,
                                       publisherAddress, pushPullAddress);
    remoteRunner.WaitSubProcessStarted();

    // Wait for devices ready notification
    while (puller.GetReadyDevice().empty()) {
        puller.Get();
    }

    // 2. Create op
    const Shape shape({3, 3});
    const DeviceMesh deviceMesh(DeviceKind::kGpu, std::vector<int64_t>({0}));
    const PlacementSeq placementSeq({Placement("R")});
    std::unique_ptr<OpParam> createParam =
        std::make_unique<CreateParam>(CreateKind::kOnes, shape, DataKind::kFloat32, deviceMesh, placementSeq);
    std::unique_ptr<Operator> op =
        OperatorFactory::GetSingleton().NewOperatorOrThrow(std::move(createParam), OperandArray());

    // 3. Execute op
    std::vector<std::shared_ptr<core::Operator>> ops;
    ops.push_back(std::move(op));

    remoteRunnerPublisher.Execute(ops, {});
}
