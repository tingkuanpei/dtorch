/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "remote_runner_subscriber.h"

#include <string>

#include "dtorch/common/filesystem.h"
#include "dtorch/external/zmq/remote_runner_publisher.h"
#include "dtorch/external/zmq/zmq.h"

namespace dtorch {
namespace external {
namespace zmq {

struct RemoteRunnerSubscriber::Impl {
    explicit Impl(const std::string& publisherAddress)
        : ctx(),
          subscriber(ctx, ::zmq::socket_type::sub),
          messageType(),
          messageId(PublishMessageIdManager::kInitValue),
          serializedData() {
        subscriber.set(::zmq::sockopt::rcvhwm, 0);
        subscriber.set(::zmq::sockopt::subscribe, "");
        subscriber.set(::zmq::sockopt::rcvtimeo, GetZmqTimeoutMilliSecond());
        // Use socket monitor to wait for the SUB connection to be fully established.
        // Must be called BEFORE connect() to capture the ZMQ_EVENT_CONNECTED event.
        // This deterministically ensures the subscriber won't miss the first published
        // messages (ZMQ SUB connect is asynchronous).
        std::string monitorAddr = "inproc://sub-monitor-" + GetRandomFileName(8);
        zmq_socket_monitor(static_cast<void*>(subscriber), monitorAddr.c_str(), ZMQ_EVENT_CONNECTED);
        ::zmq::socket_t monitorSocket(ctx, ::zmq::socket_type::pair);
        monitorSocket.connect(monitorAddr);

        subscriber.connect(publisherAddress);

        ::zmq::message_t eventMsg;
        auto recvRet = monitorSocket.recv(eventMsg);  // blocks until CONNECTED
        IgnoreUnused(recvRet);
        // monitorSocket destroyed here → monitoring stops automatically
    }

    DTORCH_DISABLE_COPY_AND_MOVE(Impl);

    ::zmq::context_t ctx;
    ::zmq::socket_t subscriber;
    std::string messageType;
    int64_t messageId;
    std::string serializedData;
};

RemoteRunnerSubscriber::RemoteRunnerSubscriber(const std::string& publisherAddress)
    : mImplPtr(std::make_shared<Impl>(publisherAddress)) {}

RemoteRunnerSubscriber::~RemoteRunnerSubscriber() = default;

bool RemoteRunnerSubscriber::Get() {
    std::vector<::zmq::message_t> buffers;
    auto ret = ::zmq::recv_multipart(mImplPtr->subscriber, std::back_inserter(buffers), ::zmq::recv_flags::dontwait);
    if (!ret.has_value()) {
        return false;
    }
    DAlwaysAssert(ret.has_value() && (ret.value() == 2 || ret.value() == 3));

    // Check message id
    int64_t gotMessageId = std::stoi(GetMsgAsString(buffers[0]));
    if (gotMessageId != mImplPtr->messageId + 1) {
        DLogFatal() << "RemoteRunnerSubscriber::Get: receive message with wrong message id: " << gotMessageId
                    << " expected: " << mImplPtr->messageId + 1;
    }

    // Store message
    mImplPtr->messageType = GetMsgAsString(buffers[1]);
    if (mImplPtr->messageType == RemoteRunnerPublisher::kExecuteStr) {
        mImplPtr->serializedData = GetMsgAsString(buffers[2]);
    }
    mImplPtr->messageId += 1;

    return true;
}

void RemoteRunnerSubscriber::Clear() {
    mImplPtr->messageType.clear();
    mImplPtr->messageId = PublishMessageIdManager::kInitValue;
    mImplPtr->serializedData.clear();
}

const std::string& RemoteRunnerSubscriber::GetMessageType() const { return mImplPtr->messageType; }

int64_t RemoteRunnerSubscriber::GetMessageId() const { return mImplPtr->messageId; }

const std::string& RemoteRunnerSubscriber::GetSerializedData() const { return mImplPtr->serializedData; }

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
