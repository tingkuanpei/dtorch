/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <string>

namespace dtorch {
namespace external {
namespace rpc {

class RpcServer {
public:
    RpcServer(const std::string& address);

    ~RpcServer();

    void BuildAndStartInBackgroundThread();

    void BuildAndStart();

    void Wait();

    void Shutdown();

public:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
