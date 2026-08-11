/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "remote_runner_pusher.h"

#include <sstream>

#include "dtorch/common/utilities.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/zmq/zmq.h"

namespace dtorch {
namespace external {
namespace zmq {

struct RemoteRunnerPusher::Impl {
    explicit Impl(const std::string& pushAddress) : ctx(), pusher(ctx, ::zmq::socket_type::push) {
        pusher.connect(pushAddress);
    }

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

    ::zmq::context_t ctx;
    ::zmq::socket_t pusher;
};

RemoteRunnerPusher::RemoteRunnerPusher(const std::string& pushAddress)
    : mImplPtr(std::make_shared<Impl>(pushAddress)) {}

RemoteRunnerPusher::~RemoteRunnerPusher() = default;

void RemoteRunnerPusher::NotifyDevicesReady(const std::vector<api::cpp::Device>& devices) {
    std::stringstream ss(std::ios::out | std::ios::binary);
    boost::BinaryOArchive boa(ss);
    boa << devices;
    std::string serializedData = ss.str();

    const std::array<::zmq::const_buffer, 2> sendMsgs = {::zmq::buffer(kDevicesReadyStr),
                                                         ::zmq::buffer(serializedData.data(), serializedData.size())};
    SendMultipart(mImplPtr->pusher, sendMsgs);
}

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
