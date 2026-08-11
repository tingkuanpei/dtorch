/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "heart_beat_interface.h"

#include "dtorch/common/logging.h"
#include "dtorch/external/rpc/rpc_server_imp.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
// Disable warning: 'class google::base::CheckOpMessageBuilder' has pointer data members
#pragma GCC diagnostic ignored "-Weffc++"
#endif
#include "dtorch/external/rpc/proto/heart_beat.grpc.pb.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include "dtorch/api/cpp/global_option.h"
#include "dtorch/core/distributed/process_heart_beat.h"

namespace dtorch {
namespace external {
namespace rpc {

// HeartBeatServer

class HeartBeatServiceImpl final : public heart_beat::HeartBeat::Service {
public:
    HeartBeatServiceImpl(core::ProcessHeartBeatBase& processHeartBeatBase)
        : mProcessHeartBeatBase(processHeartBeatBase) {}

    grpc::Status IsBeat(grpc::ServerContext* context, const heart_beat::HeartBeatRequest* request,
                        heart_beat::HeartBeatReply* reply) override {
        IgnoreUnused(context, request, reply);
        return grpc::Status::OK;
    }

    grpc::Status NotifyStopPoll(grpc::ServerContext* context, const heart_beat::HeartBeatRequest* request,
                                heart_beat::HeartBeatReply* reply) override {
        IgnoreUnused(context, request, reply);
        mProcessHeartBeatBase.NotifyStopPoll();
        return grpc::Status::OK;
    }

    grpc::Status RegisterWorkerProcess(grpc::ServerContext* context,
                                       const heart_beat::RegisterWorkerProcessRequest* request,
                                       heart_beat::RegisterWorkerProcessReply* reply) override {
        IgnoreUnused(context, request, reply);
        mProcessHeartBeatBase.RegisterWorkerProcess(request->worker_address());
        return grpc::Status::OK;
    }

    grpc::Status UnregisterWorkerProcess(grpc::ServerContext* context,
                                         const heart_beat::RegisterWorkerProcessRequest* request,
                                         heart_beat::RegisterWorkerProcessReply* reply) override {
        IgnoreUnused(context, request, reply);
        mProcessHeartBeatBase.UnregisterWorkerProcess(request->worker_address());
        return grpc::Status::OK;
    }

private:
    core::ProcessHeartBeatBase& mProcessHeartBeatBase;
};

HeartBeatServer::HeartBeatServer(const std::string& address, core::ProcessHeartBeatBase& processHeartBeatBase)
    : RpcServer(address) {
    mImplPtr->service = std::make_shared<HeartBeatServiceImpl>(processHeartBeatBase);
    BuildAndStartInBackgroundThread();
}

// HeartBeatClient

struct HeartBeatClient::Impl {
    Impl() : stub(), address() {}

    ~Impl() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

    std::unique_ptr<heart_beat::HeartBeat::Stub> stub;
    std::string address;
};

HeartBeatClient::HeartBeatClient(const std::string& address) : mImplPtr(std::make_shared<HeartBeatClient::Impl>()) {
    // clang-format off
    const std::string retryPolicy =
    "{\"methodConfig\" : [{"
    "   \"name\" : [{\"service\": \"heart_beat.HeartBeat\"}],"
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
    mImplPtr->stub =
        heart_beat::HeartBeat::NewStub(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
    mImplPtr->address = address;
};

const std::string& HeartBeatClient::GetAddress() const noexcept { return mImplPtr->address; }

bool HeartBeatClient::IsBeat() {
    heart_beat::HeartBeatRequest request;
    heart_beat::HeartBeatReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context);
    auto status = mImplPtr->stub->IsBeat(&context, request, &reply);
    return status.ok();
}

bool HeartBeatClient::NotifyStopPoll() {
    heart_beat::HeartBeatRequest request;
    heart_beat::HeartBeatReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context);
    auto status = mImplPtr->stub->NotifyStopPoll(&context, request, &reply);
    return status.ok();
}

bool HeartBeatClient::RegisterWorker(const std::string& workerAddress) {
    heart_beat::RegisterWorkerProcessRequest request;
    request.set_worker_address(workerAddress);
    heart_beat::RegisterWorkerProcessReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context);
    auto status = mImplPtr->stub->RegisterWorkerProcess(&context, request, &reply);
    return status.ok();
}

bool HeartBeatClient::UnregisterWorker(const std::string& workerAddress) {
    heart_beat::RegisterWorkerProcessRequest request;
    request.set_worker_address(workerAddress);
    heart_beat::RegisterWorkerProcessReply reply;
    grpc::ClientContext context;
    ClientContextSetDeadline(context);
    auto status = mImplPtr->stub->UnregisterWorkerProcess(&context, request, &reply);
    return status.ok();
}

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
