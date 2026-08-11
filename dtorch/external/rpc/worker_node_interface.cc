/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "worker_node_interface.h"

#include <future>

#include "dtorch/api/cpp/graph.h"
#include "dtorch/core/distributed/cluster_info.h"
#include "dtorch/core/distributed/worker_node.h"
#include "dtorch/core/runner/runner_supported_devices.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
// Disable warning: 'class google::base::CheckOpMessageBuilder' has pointer data members
#pragma GCC diagnostic ignored "-Weffc++"
#endif
#include "dtorch/external/rpc/proto/distributed_cluster_node.grpc.pb.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include "dtorch/external/rpc/rpc_server_imp.h"

namespace dtorch {
namespace external {
namespace rpc {

// WorkerNodeServer

class WorkerNodeServerImpl final : public distributed::WorkerNode::CallbackService {
public:
    WorkerNodeServerImpl(core::distributed::WorkerNode& workerNode) : mCoreWorkerNode(workerNode) {}

    grpc::ServerUnaryReactor* SendDestroySignal(grpc::CallbackServerContext* context,
                                                const distributed::EmptyRequest* request,
                                                distributed::EmptyReply* reply) override {
        IgnoreUnused(request, reply);
        mCoreWorkerNode.SendDestroySignal();
        auto* reactor = context->DefaultReactor();
        reactor->Finish(grpc::Status::OK);
        return reactor;
    }

    grpc::ServerUnaryReactor* CreateGraph(grpc::CallbackServerContext* context,
                                          const distributed::CreateGraphRequest* request,
                                          distributed::EmptyReply* reply) override {
        IgnoreUnused(reply);
        api::cpp::GraphOption graphOption = api::cpp::GraphOption::FromString(request->graph_option_data());
        core::RunnerSupportedDevices supportedDevices =
            core::RunnerSupportedDevices::FromString(request->supported_devices_data());
        core::distributed::ClusterInfo clusterInfo =
            core::distributed::ClusterInfo::FromString(request->cluster_info_data());
        mCoreWorkerNode.CreateGraph(request->graph_id(), graphOption, supportedDevices, clusterInfo,
                                    request->publisher_address(), request->push_pull_address());
        auto* reactor = context->DefaultReactor();
        reactor->Finish(grpc::Status::OK);
        return reactor;
    }

    grpc::ServerUnaryReactor* DestroyGraph(grpc::CallbackServerContext* context,
                                           const distributed::GraphIdRequest* request,
                                           distributed::EmptyReply* reply) override {
        IgnoreUnused(reply);
        mCoreWorkerNode.DestroyGraph(request->graph_id());
        auto* reactor = context->DefaultReactor();
        reactor->Finish(grpc::Status::OK);
        return reactor;
    }

private:
    core::distributed::WorkerNode& mCoreWorkerNode;
};

WorkerNodeServer::WorkerNodeServer(const std::string& address, core::distributed::WorkerNode& workerNode)
    : RpcServer(address) {
    mImplPtr->service = std::make_shared<WorkerNodeServerImpl>(workerNode);
    BuildAndStartInBackgroundThread();
}

// WorkerNodeClient

struct WorkerNodeClient::Impl {
    Impl() : stub() {}

    ~Impl() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

    std::unique_ptr<distributed::WorkerNode::Stub> stub;
};

WorkerNodeClient::WorkerNodeClient(const std::string& address) : mImplPtr(std::make_shared<WorkerNodeClient::Impl>()) {
    mImplPtr->stub = distributed::WorkerNode::NewStub(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
};

void WorkerNodeClient::SendDestroySignal() {
    distributed::EmptyRequest request;
    distributed::EmptyReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context);
    RpcCheckStatus(mImplPtr->stub->SendDestroySignal(&context, request, &reply));
}

std::future<void> WorkerNodeClient::AsyncCreateGraph(uint64_t graphId, const api::cpp::GraphOption& graphOption,
                                                     const core::RunnerSupportedDevices& supportedDevices,
                                                     const core::distributed::ClusterInfo& clusterInfo,
                                                     const std::string& publisherAddress,
                                                     const std::string& pushPullAddress,
                                                     const std::optional<double>& timeoutSecond) {
    distributed::CreateGraphRequest request;
    request.set_graph_id(graphId);
    request.set_publisher_address(publisherAddress);
    request.set_push_pull_address(pushPullAddress);
    request.set_graph_option_data(graphOption.SerializeToString());
    request.set_supported_devices_data(supportedDevices.SerializeToString());
    request.set_cluster_info_data(clusterInfo.SerializeToString());

    std::shared_ptr<grpc::ClientContext> context = std::make_shared<grpc::ClientContext>();
    // CreateGraph blocks on the WorkerNode while it spawns RemoteRunner subprocesses
    // (fork/exec + ZMQ setup), so the deadline defaults to a generous 60s (overridable).
    ClientContextSetDeadline(*context, timeoutSecond);
    std::shared_ptr<distributed::EmptyReply> reply = std::make_shared<distributed::EmptyReply>();
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();

    mImplPtr->stub->async()->CreateGraph(context.get(), &request, reply.get(),
                                         [context, reply, promise](grpc::Status status) {
                                             RpcCheckStatus(status);
                                             promise->set_value();
                                         });
    return future;
}

std::future<void> WorkerNodeClient::AsyncDestroyGraph(uint64_t graphId, const std::optional<double>& timeoutSecond) {
    distributed::GraphIdRequest request;
    request.set_graph_id(graphId);

    std::shared_ptr<grpc::ClientContext> context = std::make_shared<grpc::ClientContext>();
    ClientContextSetDeadline(*context, timeoutSecond);
    std::shared_ptr<distributed::EmptyReply> reply = std::make_shared<distributed::EmptyReply>();
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();

    mImplPtr->stub->async()->DestroyGraph(context.get(), &request, reply.get(),
                                          [context, reply, promise](grpc::Status status) {
                                              RpcCheckStatus(status);
                                              promise->set_value();
                                          });
    return future;
}

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
