/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

#ifdef __GNUC__
#pragma GCC diagnostic push
// Disable warning: 'class google::base::CheckOpMessageBuilder' has pointer data members
#pragma GCC diagnostic ignored "-Weffc++"
#endif
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"
#include "dtorch/external/rpc/rpc_common.h"
#include "dtorch/external/rpc/rpc_server.h"

namespace dtorch {
namespace external {
namespace rpc {

void RpcCheckStatus(const grpc::Status& status);

void ClientContextSetDeadline(grpc::ClientContext& context, const std::optional<double>& timeoutSecond = std::nullopt);

struct RpcServer::Impl {
    Impl() : fileRemoveGuard(), backgroundThread(), service(), builder(), server() {}

    ~Impl() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

    std::unique_ptr<FileRemoveGuard> fileRemoveGuard;
    std::unique_ptr<std::thread> backgroundThread;
    std::shared_ptr<grpc::Service> service;
    grpc::ServerBuilder builder;
    std::unique_ptr<grpc::Server> server;
};

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
