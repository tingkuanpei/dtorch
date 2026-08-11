/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "memory_tensor_promise_future.h"

#include <chrono>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace core {
namespace communication {

// ============================================================
// MemoryTensorFuture
// ============================================================

MemoryTensorFuture::MemoryTensorFuture(std::future<std::shared_ptr<torch::Tensor>> future)
    : TensorFuture(), mFuture(std::move(future)) {}

std::shared_ptr<torch::Tensor> MemoryTensorFuture::Get() { return mFuture.get(); }

void MemoryTensorFuture::Wait() { mFuture.wait(); }

bool MemoryTensorFuture::WaitFor(int64_t timeoutMs) {
    return mFuture.wait_for(std::chrono::milliseconds(timeoutMs)) == std::future_status::ready;
}

bool MemoryTensorFuture::IsReady() const {
    return mFuture.valid() && mFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// ============================================================
// MemoryTensorPromise
// ============================================================

MemoryTensorPromise::MemoryTensorPromise() : TensorPromise(), mPromise() {}

void MemoryTensorPromise::SetValue(std::shared_ptr<torch::Tensor> tensor) { mPromise.set_value(tensor); }

std::unique_ptr<TensorFuture> MemoryTensorPromise::GetFuture() {
    try {
        return std::make_unique<MemoryTensorFuture>(mPromise.get_future());
    } catch (const std::future_error& e) {
        DLogFatal() << "MemoryTensorPromise::GetFuture() failed: " << e.what()
                    << ". get_future() can only be called once.";
    }
    return nullptr;
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
