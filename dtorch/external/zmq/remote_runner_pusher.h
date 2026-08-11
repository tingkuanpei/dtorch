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

class RemoteRunnerPusher {
public:
    inline static const std::string kDevicesReadyStr = "devicesReady";

public:
    explicit RemoteRunnerPusher(const std::string& pushAddress);

    ~RemoteRunnerPusher();

    void NotifyDevicesReady(const std::vector<api::cpp::Device>& devices);

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

}  // namespace zmq
}  // namespace external
}  // namespace dtorch
