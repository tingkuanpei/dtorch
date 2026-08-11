/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dtorch/api/cpp/device.h"

namespace dtorch {
namespace external {
namespace zmq {

class RemoteRunnerPuller {
public:
    explicit RemoteRunnerPuller(const std::string& pushAddress);

    ~RemoteRunnerPuller();

    // Tries to receive one message via non-blocking recv_multipart.
    // Returns true if a new message was received; false if no message is available.
    // On receive, updates internal messageType and readyDevices.
    // On no-receive, leaves internal state unchanged (cached values persist).
    bool Get();

    // Resets stored message state to initial values.
    void Clear();

    // Returns the message type of the last received message.
    // Empty string if no message has been received yet or after Clear().
    const std::string& GetMessageType() const;

    // Returns the ready device list of the last received message.
    // Valid only when GetMessageType() == RemoteRunnerPusher::kDevicesReadyStr.
    // Empty vector for no-message state or after Clear().
    const std::vector<api::cpp::Device>& GetReadyDevice() const;

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
