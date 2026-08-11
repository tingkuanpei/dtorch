/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "memory_void_promise_future.h"

#include <chrono>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

namespace dtorch {
namespace core {
namespace communication {

// ============================================================
// MemoryVoidFuture
// ============================================================

MemoryVoidFuture::MemoryVoidFuture(std::future<void> future) : VoidFuture(), mFuture(std::move(future)) {}

void MemoryVoidFuture::Get() { mFuture.get(); }

void MemoryVoidFuture::Wait() { mFuture.wait(); }

bool MemoryVoidFuture::WaitFor(int64_t timeoutMs) {
    auto status = mFuture.wait_for(std::chrono::milliseconds(timeoutMs));
    if (status == std::future_status::ready) {
        mFuture.get();
        return true;
    }
    return false;
}

bool MemoryVoidFuture::IsReady() const {
    return mFuture.valid() && mFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// ============================================================
// MemoryVoidPromise
// ============================================================

MemoryVoidPromise::MemoryVoidPromise() : VoidPromise(), mPromise() {}

void MemoryVoidPromise::SetValue() { mPromise.set_value(); }

std::unique_ptr<VoidFuture> MemoryVoidPromise::GetFuture() {
    try {
        return std::make_unique<MemoryVoidFuture>(mPromise.get_future());
    } catch (const std::future_error& e) {
        DLogFatal() << "MemoryVoidPromise::GetFuture() failed: " << e.what()
                    << ". get_future() can only be called once.";
    }
    return nullptr;
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
