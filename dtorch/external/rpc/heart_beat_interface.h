/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <string>

#include "dtorch/external/rpc/rpc_server.h"

namespace dtorch {
namespace core {
class ProcessHeartBeatBase;
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace external {
namespace rpc {

class HeartBeatServer : public RpcServer {
public:
    HeartBeatServer(const std::string& address, core::ProcessHeartBeatBase& processHeartBeatBase);
};

class HeartBeatClient {
public:
    HeartBeatClient(const std::string& address);

    const std::string& GetAddress() const noexcept;

    bool IsBeat();

    bool NotifyStopPoll();

    bool RegisterWorker(const std::string& workerAddress);

    bool UnregisterWorker(const std::string& workerAddress);

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
