/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace dtorch {
namespace external {
namespace zmq {

class RemoteRunnerSubscriber {
public:
    explicit RemoteRunnerSubscriber(const std::string& publisherAddress);

    ~RemoteRunnerSubscriber();

    // Tries to receive one message via non-blocking recv_multipart.
    // Returns true if a new message was received; false if no message is available.
    // On receive, updates internal messageType, messageId, and serializedData.
    // On no-receive, leaves internal state unchanged (cached values persist).
    bool Get();

    // Resets stored message state to initial values.
    void Clear();

    // Returns the message type of the last received message.
    // Empty string if no message has been received yet or after Clear().
    const std::string& GetMessageType() const;

    // Returns the message ID of the last received message.
    // PublishMessageIdManager::kInitValue (-1) if no message received yet or after Clear().
    int64_t GetMessageId() const;

    // Returns serialized data of the last received execute message.
    // Valid only when GetMessageType() == RemoteRunnerPublisher::kExecuteStr.
    // Empty string for destroy messages, no-message state, or after Clear().
    const std::string& GetSerializedData() const;

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
