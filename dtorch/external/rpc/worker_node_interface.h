/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>

#include "dtorch/external/rpc/rpc_server.h"

namespace dtorch {
namespace api {
namespace cpp {
struct GraphOption;
}  // namespace cpp
}  // namespace api
namespace core {
class RunnerSupportedDevices;
namespace distributed {
class WorkerNode;
class ClusterInfo;
}  // namespace distributed
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace external {
namespace rpc {

class WorkerNodeServer : public RpcServer {
public:
    WorkerNodeServer(const std::string& address, core::distributed::WorkerNode& workerNode);
};

class WorkerNodeClient {
public:
    WorkerNodeClient(const std::string& address);

    void SendDestroySignal();

    // CreateGraph runs long: each WorkerNode forks/execs its RemoteRunner subprocesses and waits for
    // their ZMQ setup to finish before replying. That far exceeds the short global gRPC timeout
    // (DTORCH_GRPC_TIMEOUT_SECOND, ~3s), so it cannot use the global default and instead defaults to a
    // 60s deadline. Pass timeoutSecond to override (std::nullopt falls back to the global timeout).
    std::future<void> AsyncCreateGraph(uint64_t graphId, const api::cpp::GraphOption& graphOption,
                                       const core::RunnerSupportedDevices& supportedDevices,
                                       const core::distributed::ClusterInfo& clusterInfo,
                                       const std::string& publisherAddress, const std::string& pushPullAddress,
                                       const std::optional<double>& timeoutSecond = 60.0);

    // DestroyGraph is symmetric: tearing down a WorkerNode's RemoteRunner subprocesses before replying
    // can also be slow, so it likewise cannot use the short global gRPC timeout and defaults to 60s.
    std::future<void> AsyncDestroyGraph(uint64_t graphId, const std::optional<double>& timeoutSecond = 60.0);

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
