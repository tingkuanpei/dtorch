/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include "dtorch/api/cpp/cluster.h"
#include "dtorch/api/cpp/dtorch.h"
#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/ip_address.h"
#include "dtorch/common/process/sub_process.h"
#include "dtorch/core/distributed/main_node.h"
#include "test.h"

namespace dtorch {
namespace {

using api::cpp::distributed::Cluster;
using api::cpp::distributed::MainNode;
using api::cpp::distributed::WorkerNode;

// ===== Address validation tests =====

TEST(AddressTest, CheckStringIsValidAddress) {
    // Valid addresses
    EXPECT_TRUE(CheckStringIsValidAddress("127.0.0.1:13000"));
    EXPECT_TRUE(CheckStringIsValidAddress("192.168.1.1:50051"));
    EXPECT_TRUE(CheckStringIsValidAddress("0.0.0.0:1"));
    EXPECT_TRUE(CheckStringIsValidAddress("10.0.0.1:65535"));

    // Invalid: missing port separator
    EXPECT_FALSE(CheckStringIsValidAddress("127.0.0.1"));
    EXPECT_FALSE(CheckStringIsValidAddress(""));
    EXPECT_FALSE(CheckStringIsValidAddress(":13000"));
    EXPECT_FALSE(CheckStringIsValidAddress("127.0.0.1:"));

    // Invalid IP
    EXPECT_FALSE(CheckStringIsValidAddress("invalid:13000"));
    EXPECT_FALSE(CheckStringIsValidAddress("999.999.999.999:13000"));
    EXPECT_FALSE(CheckStringIsValidAddress("hostname:13000"));

    // Invalid port
    EXPECT_FALSE(CheckStringIsValidAddress("127.0.0.1:0"));
    EXPECT_FALSE(CheckStringIsValidAddress("127.0.0.1:65536"));
    EXPECT_FALSE(CheckStringIsValidAddress("127.0.0.1:-1"));
    EXPECT_FALSE(CheckStringIsValidAddress("127.0.0.1:abc"));
}

// ===== MainNode tests =====

class MainNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure clean state: stop any leftover cluster from other tests
        if (Cluster::GetSingleton().IsCreate()) {
            if (Cluster::GetSingleton().GetNodeType() == api::cpp::distributed::NodeType::kMain) {
                MainNode::Stop();
            }
        }
    }

    void TearDown() override {
        if (Cluster::GetSingleton().IsCreate()) {
            if (Cluster::GetSingleton().GetNodeType() == api::cpp::distributed::NodeType::kMain) {
                MainNode::Stop();
            }
        }
    }
};

TEST_F(MainNodeTest, SetMainNodeAddressValid) {
    MainNode::SetMainNodeAddress("127.0.0.1:13000");
    EXPECT_EQ(core::distributed::MainNode::GetRefMainNodeAddress(), "127.0.0.1:13000");
}

TEST_F(MainNodeTest, SetMainNodeAddressInvalid) {
    EXPECT_THROW(MainNode::SetMainNodeAddress("invalid_address"), std::invalid_argument);
}

TEST_F(MainNodeTest, SetMainNodeAddressAfterStart) {
    MainNode::SetMainNodeAddress("127.0.0.1:13000");
    EXPECT_TRUE(MainNode::Start());
    EXPECT_THROW(MainNode::SetMainNodeAddress("192.168.1.1:13000"), std::invalid_argument);
}

TEST_F(MainNodeTest, StartAndStop) {
    MainNode::SetMainNodeAddress("127.0.0.1:13000");
    EXPECT_TRUE(MainNode::Start());
    EXPECT_TRUE(Cluster::GetSingleton().IsCreate());
    EXPECT_EQ(Cluster::GetSingleton().GetNodeType(), api::cpp::distributed::NodeType::kMain);

    MainNode::Stop();
    EXPECT_FALSE(Cluster::GetSingleton().IsCreate());
}

TEST_F(MainNodeTest, NumNodeInCluster) {
    MainNode::SetMainNodeAddress("127.0.0.1:13000");
    EXPECT_TRUE(MainNode::Start());
    // Main node itself counts as 1 node in the cluster
    EXPECT_EQ(MainNode::NumNodeInCluster(), 1);
}

TEST_F(MainNodeTest, GetMainNodeAddress) {
    MainNode::SetMainNodeAddress("127.0.0.1:13000");
    EXPECT_TRUE(MainNode::Start());
    EXPECT_EQ(MainNode::GetMainNodeAddress(), "127.0.0.1:13000");
}

TEST_F(MainNodeTest, WaitClusterReadySingleNode) {
    MainNode::SetMainNodeAddress("127.0.0.1:13000");
    EXPECT_TRUE(MainNode::Start());
    // Main node itself is already registered, so WaitClusterReady(1) should return true immediately
    EXPECT_TRUE(MainNode::WaitClusterReady(1, 1.0));
}

TEST_F(MainNodeTest, WaitClusterReadyTimeout) {
    MainNode::SetMainNodeAddress("127.0.0.1:13000");
    EXPECT_TRUE(MainNode::Start());
    // WaitClusterReady(2) should timeout since no worker nodes will join
    EXPECT_FALSE(MainNode::WaitClusterReady(2, 0.1));
}

// ===== WorkerNode tests =====
// WorkerNode::Start is private; use fork + ExecMain (public API).
// SubProcess is not suitable for failure cases because its destructor asserts exit_code == 0.

TEST(WorkerNodeTest, ExecMainFailsWithoutMainNode) {
    std::string workerAddr = GetValidNodeAddress(16000, 16100);

    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if (pid == 0) {
        std::vector<std::string> args = {"dtorch_launcher", "--main-node-address=127.0.0.1:19999",
                                         "--this-node-address=" + workerAddr, "--timeout-second=0.5"};
        _exit(WorkerNode::ExecMain(args));
    }

    int status;
    waitpid(pid, &status, 0);
    // Without MainNode running, ExecMain calls RpcCheckStatus which calls DLogFatal (SIGABRT)
    EXPECT_FALSE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// ===== MainNode + WorkerNode integration test =====

TEST(ClusterIntegrationTest, MainNodeWithWorkerNode) {
    if (core::GlobalOption::GetSingleton().GetDTensorInSameDevice()) {
        GTEST_SKIP() << "DTORCH_DTENSOR_IN_SAME_DEVICE is enabled — worker nodes are not supported in same-device mode";
    }

    std::string mainAddr = GetValidNodeAddress(14000, 14100);
    std::string workerAddr = GetValidNodeAddress(14100, 14200);

    MainNode::SetMainNodeAddress(mainAddr);
    ASSERT_TRUE(MainNode::Start());

    // Launch WorkerNode via SubProcess — uses dtorch_launcher script
    std::string cmd =
        "dtorch_launcher"
        " --main-node-address=" +
        mainAddr + " --this-node-address=" + workerAddr + " --timeout-second=10";
    SubProcess workerProcess(cmd);

    // Wait for WorkerNode to register — cluster should now have 2 nodes
    EXPECT_TRUE(MainNode::WaitClusterReady(2, 15.0));
    EXPECT_EQ(MainNode::NumNodeInCluster(), 2);

    // Stop MainNode — ~MainNode sends DestroySignal to worker
    MainNode::Stop();
    // SubProcess destructor waits for worker to exit and asserts exit_code == 0
}

// ===== MainNode + WorkerNode distributed computation test =====
// Create both MainNode and WorkerNode on the same machine to simulate the computation of two machines.
TEST(ClusterIntegrationTest, MainNodeWorkerNodeDistributedAdd) {
    if (core::GlobalOption::GetSingleton().GetDTensorInSameDevice()) {
        GTEST_SKIP() << "DTORCH_DTENSOR_IN_SAME_DEVICE is enabled — worker nodes are not supported in same-device mode";
    }
    if (!api::cpp::Device::IsAvailable(api::cpp::DeviceKind::kGpu)) {
        GTEST_SKIP() << "Requires CUDA (MainNode + WorkerNode share the single GPU)";
    }

    std::string mainAddr = GetValidNodeAddress(14000, 14100);
    std::string workerAddr = GetValidNodeAddress(14100, 14200);

    MainNode::SetMainNodeAddress(mainAddr);
    ASSERT_TRUE(MainNode::Start());

    // Launch WorkerNode via SubProcess. Both processes see the single GPU; the WorkerNode reports
    // gpu_count=1, so the cluster has global cuda 0 (MainNode) and global cuda 1 (WorkerNode).
    std::string cmd = "dtorch_launcher --main-node-address=" + mainAddr + " --this-node-address=" + workerAddr +
                      " --timeout-second=60";
    SubProcess workerProcess(cmd);

    ASSERT_TRUE(MainNode::WaitClusterReady(2, 30.0));
    EXPECT_EQ(MainNode::NumNodeInCluster(), 2);

    api::cpp::GraphOption option;
    option.perDevicePerProcess = false;
    api::cpp::Graph graph(option);

    api::cpp::DeviceMesh deviceMesh(api::cpp::DeviceKind::kGpu, {0, 1});
    ASSERT_TRUE(graph.Satisfy(deviceMesh));

    // TODO: The tensor operations below require TensorStoreType::kNetwork support before they can
    // work with perDevicePerProcess=false in a cluster. Currently PerDeviceThreadNodeRunner always
    // initializes its NaiveRunner with TensorStoreType::kMemory (in-process only), which cannot
    // exchange tensor data with the WorkerNode's RemoteRunner subprocess (which uses kFile via IPC).
    // Once kNetwork is implemented, uncomment the placement, tensor construction, _Add, and
    // GetTorchTensor blocks below to validate the full distributed computation.

    // api::cpp::PlacementSeq placementSeq = {api::cpp::Shard(0)};

    // torch::Tensor torchA = torch::rand({4}).to(torch::kCUDA);
    // torch::Tensor torchB = torch::rand({4}).to(torch::kCUDA);
    // torch::Tensor torchC = torchA.add(torchB);

    // api::cpp::Tensor tensorA(graph, torchA, deviceMesh, placementSeq);
    // api::cpp::Tensor tensorB(graph, torchB, deviceMesh, placementSeq);
    // api::cpp::Tensor tensorC = api::cpp::functional::_Add(tensorA, tensorB);
    // torch::Tensor fromC = tensorC.GetTorchTensor();

    // EXPECT_TRUE(torch::allclose(torchC, fromC));

    graph.Destroy();
    MainNode::Stop();
    // SubProcess destructor waits for the worker to exit and asserts exit_code == 0.
}

}  // namespace
}  // namespace dtorch
