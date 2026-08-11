/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <string>
#include <thread>

#include "dtorch/common/logging.h"
#include "dtorch/external/rpc/greeter_interface.h"
#include "dtorch/external/rpc/rpc_common.h"
#include "test.h"

using dtorch::external::rpc::GetRandomUdsAddress;
using dtorch::external::rpc::GreeterClient;
using dtorch::external::rpc::GreeterServer;

void TestImp(const std::string& address) {
    GreeterServer server(address);

    GreeterClient client(address);
    auto syncReplyStr = client.SyncSayGreeter("World");
    EXPECT_TRUE(syncReplyStr == "Hello World");
    auto asyncReplyStr = client.AsyncSayGreeter("World");
    EXPECT_TRUE(asyncReplyStr.get() == "Hello World");
}

TEST(RpcTest, SimpleTest) {
    TestImp("localhost:50051");
    TestImp(GetRandomUdsAddress());
}
