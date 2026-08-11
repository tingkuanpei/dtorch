/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "remote_runner_puller.h"

#include <sstream>

#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/zmq/remote_runner_pusher.h"
#include "dtorch/external/zmq/zmq.h"

namespace dtorch {
namespace external {
namespace zmq {

struct RemoteRunnerPuller::Impl {
    explicit Impl(const std::string& pushAddress)
        : ctx(), puller(ctx, ::zmq::socket_type::pull), messageType(), readyDevices() {
        puller.set(::zmq::sockopt::rcvtimeo, GetZmqTimeoutMilliSecond());
        puller.bind(pushAddress);
    }

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

    ::zmq::context_t ctx;
    ::zmq::socket_t puller;
    std::string messageType;
    std::vector<api::cpp::Device> readyDevices;
};

RemoteRunnerPuller::RemoteRunnerPuller(const std::string& pushAddress)
    : mImplPtr(std::make_shared<Impl>(pushAddress)) {}

RemoteRunnerPuller::~RemoteRunnerPuller() = default;

bool RemoteRunnerPuller::Get() {
    std::vector<::zmq::message_t> buffers;
    auto ret = ::zmq::recv_multipart(mImplPtr->puller, std::back_inserter(buffers), ::zmq::recv_flags::dontwait);
    if (!ret.has_value()) {
        return false;
    }
    DAlwaysAssert(ret.has_value() && ret.value() == 2);

    mImplPtr->messageType = GetMsgAsString(buffers[0]);

    std::string serializedData = GetMsgAsString(buffers[1]);
    std::stringstream ss(serializedData, std::ios::in | std::ios::binary);
    boost::BinaryIArchive bia(ss);
    bia >> mImplPtr->readyDevices;

    return true;
}

void RemoteRunnerPuller::Clear() {
    mImplPtr->messageType.clear();
    mImplPtr->readyDevices.clear();
}

const std::string& RemoteRunnerPuller::GetMessageType() const { return mImplPtr->messageType; }

const std::vector<api::cpp::Device>& RemoteRunnerPuller::GetReadyDevice() const { return mImplPtr->readyDevices; }

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
