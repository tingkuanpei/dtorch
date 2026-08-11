/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "dtorch/external/rpc/rpc_server.h"

namespace dtorch {
namespace core {
namespace distributed {
class MainNode;
}  // namespace distributed
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace external {
namespace rpc {

class MainNodeServer : public RpcServer {
public:
    MainNodeServer(const std::string& address, core::distributed::MainNode& mainNode);
};

class MainNodeClient {
public:
    MainNodeClient(const std::string& address, double timeoutSecond);

    void RegisterWorkerNode(const std::string& workerNodeAddress, int64_t gpuCount,
                            const std::optional<double>& timeoutSecond = std::nullopt);

    // Sync GlobalOption from MainNode. Returns serialized data for InitFromMainNode.
    std::string SyncGlobalOption(const std::optional<double>& timeoutSecond = std::nullopt);

    // Sync environment variables from MainNode.
    // std::nullopt value means the variable is not defined on MainNode (caller should unsetenv).
    std::vector<std::pair<std::string, std::optional<std::string>>> SyncEnvVars(
        const std::optional<double>& timeoutSecond = std::nullopt);

    // Sync cluster config (heartbeat address, global comm instance id) from MainNode.
    std::pair<std::string, std::string> SyncClusterConfig(const std::optional<double>& timeoutSecond = std::nullopt);

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
