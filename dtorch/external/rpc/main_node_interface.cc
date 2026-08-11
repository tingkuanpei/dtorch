/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "main_node_interface.h"

#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/environment_variable.h"
#include "dtorch/core/communication/global_instance_id.h"
#include "dtorch/core/distributed/main_node.h"
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

// MainNodeServer

class MainNodeServerImpl final : public distributed::MainNode::Service {
public:
    MainNodeServerImpl(core::distributed::MainNode& mainNode) : mCoreMainNode(mainNode) {}

    grpc::Status RegisterWorkerNode(grpc::ServerContext* context, const distributed::RegisterWorkerNodeRequest* request,
                                    distributed::EmptyReply* reply) override {
        IgnoreUnused(context, reply);
        try {
            mCoreMainNode.AddWorkerNode(request->worker_node_address(), request->gpu_count());
        } catch (const std::exception& e) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
        }
        return grpc::Status::OK;
    }

    grpc::Status SyncGlobalOption(grpc::ServerContext* context, const distributed::EmptyRequest* request,
                                  distributed::SyncGlobalOptionReply* reply) override {
        IgnoreUnused(context, request);
        reply->set_global_option_data(core::GlobalOption::GetSingleton().SerializeToString());
        return grpc::Status::OK;
    }

    grpc::Status SyncEnvVars(grpc::ServerContext* context, const distributed::EmptyRequest* request,
                             distributed::SyncEnvVarsReply* reply) override {
        IgnoreUnused(context, request);
        for (const auto& name : core::distributed::MainNode::GetEnvVarsToSync()) {
            auto* envVar = reply->add_env_vars();
            envVar->set_key(name);
            auto value = dtorch::GetEnv(name);
            if (value) {
                envVar->set_has_value(true);
                envVar->set_value(std::string(value.get()));
            } else {
                envVar->set_has_value(false);
            }
        }
        return grpc::Status::OK;
    }

    grpc::Status SyncClusterConfig(grpc::ServerContext* context, const distributed::EmptyRequest* request,
                                   distributed::SyncClusterConfigReply* reply) override {
        IgnoreUnused(context, request);
        reply->set_main_process_heart_beat_address(mCoreMainNode.GetMainProcessHeartBeatAddress());
        reply->set_global_comm_instance_id(core::communication::GlobalCommInstanceId::GetSingleton().GetInstanceId());
        return grpc::Status::OK;
    }

private:
    core::distributed::MainNode& mCoreMainNode;
};

MainNodeServer::MainNodeServer(const std::string& address, core::distributed::MainNode& mainNode) : RpcServer(address) {
    mImplPtr->service = std::make_shared<MainNodeServerImpl>(mainNode);
    BuildAndStartInBackgroundThread();
}

// MainNodeClient

struct MainNodeClient::Impl {
    Impl() : channel(), stub() {}

    ~Impl() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<distributed::MainNode::Stub> stub;
};

MainNodeClient::MainNodeClient(const std::string& address, double timeoutSecond)
    : mImplPtr(std::make_shared<MainNodeClient::Impl>()) {
    // clang-format off
    const std::string retryPolicy =
    "{\"methodConfig\" : [{"
    "   \"name\" : [{\"service\": \"distributed.MainNode\"}],"
    "   \"waitForReady\": true,"
    "   \"retryPolicy\": {"
    "     \"maxAttempts\": 4,"
    "     \"initialBackoff\": \"1s\","
    "     \"maxBackoff\": \"5s\","
    "     \"backoffMultiplier\": 2.0,"
    "     \"retryableStatusCodes\": [\"UNAVAILABLE\"]"
    "    }"
    "}]}";
    // clang-format on

    auto args = grpc::ChannelArguments();
    args.SetServiceConfigJSON(std::string(retryPolicy));
    mImplPtr->channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args);

    // Check if the channel can connect to the main node
    auto deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(static_cast<int>(timeoutSecond * 1000));
    bool connected = mImplPtr->channel->WaitForConnected(deadline);
    if (!connected) {
        throw std::runtime_error(
            "Failed to connect to MainNode at " + address + " within " + std::to_string(timeoutSecond) +
            " seconds, channel state: " + std::to_string(static_cast<int>(mImplPtr->channel->GetState(false))));
    }

    mImplPtr->stub = distributed::MainNode::NewStub(mImplPtr->channel);
};

void MainNodeClient::RegisterWorkerNode(const std::string& workerNodeAddress, int64_t gpuCount,
                                        const std::optional<double>& timeoutSecond) {
    distributed::RegisterWorkerNodeRequest request;
    request.set_worker_node_address(workerNodeAddress);
    request.set_gpu_count(gpuCount);
    distributed::EmptyReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context, timeoutSecond);

    RpcCheckStatus(mImplPtr->stub->RegisterWorkerNode(&context, request, &reply));
}

std::string MainNodeClient::SyncGlobalOption(const std::optional<double>& timeoutSecond) {
    distributed::EmptyRequest request;
    distributed::SyncGlobalOptionReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context, timeoutSecond);
    RpcCheckStatus(mImplPtr->stub->SyncGlobalOption(&context, request, &reply));
    return reply.global_option_data();
}

std::vector<std::pair<std::string, std::optional<std::string>>> MainNodeClient::SyncEnvVars(
    const std::optional<double>& timeoutSecond) {
    distributed::EmptyRequest request;
    distributed::SyncEnvVarsReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context, timeoutSecond);
    RpcCheckStatus(mImplPtr->stub->SyncEnvVars(&context, request, &reply));

    std::vector<std::pair<std::string, std::optional<std::string>>> envVars;
    envVars.reserve(reply.env_vars_size());
    for (const auto& envVar : reply.env_vars()) {
        if (envVar.has_value()) {
            envVars.emplace_back(envVar.key(), std::optional<std::string>(envVar.value()));
        } else {
            envVars.emplace_back(envVar.key(), std::nullopt);
        }
    }
    return envVars;
}

std::pair<std::string, std::string> MainNodeClient::SyncClusterConfig(const std::optional<double>& timeoutSecond) {
    distributed::EmptyRequest request;
    distributed::SyncClusterConfigReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context, timeoutSecond);
    RpcCheckStatus(mImplPtr->stub->SyncClusterConfig(&context, request, &reply));
    return {reply.main_process_heart_beat_address(), reply.global_comm_instance_id()};
}

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
