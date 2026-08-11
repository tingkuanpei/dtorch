/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
// Disable warning: 'class google::base::CheckOpMessageBuilder' has pointer data members
#pragma GCC diagnostic ignored "-Weffc++"
#endif
#include <zmq_addon.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/filesystem.h"

namespace dtorch {
namespace external {
namespace zmq {

DTORCH_FORCEINLINE int GetZmqTimeoutMilliSecond() {
    return static_cast<int>(core::GlobalOption::GetSingleton().GetZmqTimeoutSecond() * 1000);
}

DTORCH_FORCEINLINE void CheckSendResult(const ::zmq::send_result_t& ret, size_t expected_count = 1) {
    DAlwaysAssert(ret.has_value());
    DAlwaysAssert(ret.value() == expected_count);
}

DTORCH_FORCEINLINE void CheckRecvResult(const ::zmq::recv_result_t& ret, size_t expected_count = 1) {
    DAlwaysAssert(ret.has_value());
    DAlwaysAssert(ret.value() == expected_count);
}

template <size_t N>
DTORCH_FORCEINLINE void SendMultipart(::zmq::socket_t& socket, const std::array<::zmq::const_buffer, N>& buffers) {
    CheckSendResult(::zmq::send_multipart(socket, buffers), buffers.size());
}

DTORCH_FORCEINLINE void RecvMultipart(::zmq::socket_t& socket, std::vector<::zmq::message_t>& buffers,
                                      size_t expectedCount = 1, bool waitUntilRecv = false) {
    ::zmq::recv_result_t ret;
    while (true) {
        ret = ::zmq::recv_multipart(socket, std::back_inserter(buffers));
        if (waitUntilRecv && !ret.has_value()) {
            continue;
        } else {
            break;
        }
    }
    CheckRecvResult(ret, expectedCount);
}

DTORCH_FORCEINLINE const std::string GetMsgAsString(const ::zmq::message_t& msg) {
    return std::string(static_cast<const char*>(msg.data()), msg.size());
}

DTORCH_FORCEINLINE const std::string GetRandomZmqIpcAddress(size_t length = 16) {
    return "ipc://" + GetTempDirectoryPath() + "/DTorch_ZMQ_IPC_" + GetRandomFileName(length) + ".sock";
}

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
