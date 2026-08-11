/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

#include <torch/torch.h>

#include "api_type.h"

namespace dtorch {
namespace core {
namespace communication {
class TensorFuture;
}  // namespace communication
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace api {
namespace cpp {

class Tensor;

class TensorFuture {
public:
    // Construct from internal future implementation
    explicit TensorFuture(const Tensor& tensor, std::unique_ptr<core::communication::TensorFuture> futureImpl);

    ~TensorFuture();

    // Blocking get Tensor value (waits indefinitely until ready)
    torch::Tensor Get();

    // Blocking wait (like std::future::wait, returns void)
    void Wait();

    // Wait with timeout (milliseconds). Returns true if ready, false on timeout.
    bool WaitFor(int64_t timeoutMs);

    // Check if the tensor value is ready (non-blocking).
    bool IsReady() const;

private:
    struct Impl;
    std::shared_ptr<Impl> mImplPtr;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
