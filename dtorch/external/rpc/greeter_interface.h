/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <future>
#include <memory>
#include <string>

#include "dtorch/external/rpc/rpc_server.h"

namespace dtorch {
namespace external {
namespace rpc {

class GreeterServer : public RpcServer {
public:
    GreeterServer(const std::string& address);
};

class GreeterClient {
public:
    GreeterClient(const std::string& address);

    std::string SyncSayGreeter(const std::string& user);

    std::future<std::string> AsyncSayGreeter(const std::string& user);

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
