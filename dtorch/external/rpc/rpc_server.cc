/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "rpc_server.h"

#include "dtorch/api/cpp/global_option.h"
#include "rpc_server_imp.h"

namespace dtorch {
namespace external {
namespace rpc {

void RpcCheckStatus(const grpc::Status& status) {
    if (!status.ok()) {
        DLogFatal() << "RPC error: " << status.error_code() << ": " << status.error_message();
    }
}

void ClientContextSetDeadline(grpc::ClientContext& context, const std::optional<double>& timeoutSecond) {
    if (timeoutSecond.has_value()) {
        context.set_deadline(std::chrono::system_clock::now() +
                             std::chrono::milliseconds(static_cast<int>(timeoutSecond.value() * 1000)));
    } else {
        context.set_deadline(std::chrono::system_clock::now() +
                             std::chrono::seconds(core::GlobalOption::GetSingleton().GetGrpcTimeoutSecond()));
    }
}

RpcServer::RpcServer(const std::string& address) : mImplPtr(std::make_shared<RpcServer::Impl>()) {
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    mImplPtr->fileRemoveGuard = GetUdsFileRemoveGuard(address);
    mImplPtr->builder.AddListeningPort(address, grpc::InsecureServerCredentials());
}

RpcServer::~RpcServer() {
    this->Shutdown();

    if (mImplPtr->backgroundThread) {
        mImplPtr->backgroundThread->join();
    }
}

void RpcServer::BuildAndStartInBackgroundThread() {
    DAlwaysAssert(mImplPtr->service != nullptr);
    mImplPtr->builder.RegisterService(mImplPtr->service.get());

    std::atomic_bool serverRuned = false;
    auto ServerFunc = [this, &serverRuned]() {
        DDebugAssert(this->mImplPtr != nullptr);
        this->mImplPtr->server = this->mImplPtr->builder.BuildAndStart();
        serverRuned = true;
        this->mImplPtr->server->Wait();
    };

    mImplPtr->backgroundThread = std::make_unique<std::thread>(ServerFunc);

    while (!serverRuned) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RpcServer::BuildAndStart() {
    DAlwaysAssert(mImplPtr->service != nullptr);
    mImplPtr->builder.RegisterService(mImplPtr->service.get());
    this->mImplPtr->server = this->mImplPtr->builder.BuildAndStart();
}

void RpcServer::Wait() {
    DAlwaysAssert(mImplPtr->service != nullptr);
    this->mImplPtr->server->Wait();
}

void RpcServer::Shutdown() {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->server->Shutdown();
}

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
