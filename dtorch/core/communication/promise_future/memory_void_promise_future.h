/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <future>
#include <memory>

#include "dtorch/common/debug.h"
#include "void_promise_future.h"

namespace dtorch {
namespace core {
namespace communication {

// ============================================================
// class MemoryVoidFuture — std::future based Future
// ============================================================
//
// 基于 std::future<void> 实现。
// 内部持有一个 std::future，Wait() 调用 std::future::get() 阻塞等待，
// IsReady() 通过 wait_for(0ms) 非阻塞检查 ready 状态。
// 适用于单机多线程场景（同进程内传递），零额外开销。

class MemoryVoidFuture : public VoidFuture {
public:
    explicit MemoryVoidFuture(std::future<void> future);

    ~MemoryVoidFuture() override = default;

    DTORCH_DISABLE_COPY_AND_MOVE(MemoryVoidFuture);

    void Get() override;

    void Wait() override;

    bool WaitFor(int64_t timeoutMs) override;

    bool IsReady() const override;

private:
    std::future<void> mFuture;
};

// ============================================================
// class MemoryVoidPromise — std::promise based Promise
// ============================================================
//
// 基于 std::promise<void> 实现。
// SetValue() 调用 std::promise::set_value() 发出完成信号，
// GetFuture() 调用 std::promise::get_future() 创建关联的 MemoryVoidFuture
//   （仅可调用一次，重复调用时捕获 std::future_error 后 DLogFatal）。
// Serialize() / Deserialize() 不支持（同进程内无需序列化）。
// 适用于单机多线程场景，零额外开销。

class MemoryVoidPromise : public VoidPromise {
public:
    MemoryVoidPromise();

    ~MemoryVoidPromise() override = default;

    DTORCH_DISABLE_COPY_AND_MOVE(MemoryVoidPromise);

    void SetValue() override;

    std::unique_ptr<VoidFuture> GetFuture() override;

    VoidPromiseType GetType() const override { return VoidPromiseType::kMemory; }

    std::string Serialize() const override { return ""; }

    void Deserialize(const std::string& data) override {
        IgnoreUnused(data);
        DDebugAssert(false && "MemoryVoidPromise does not support deserialization");
    }

private:
    std::promise<void> mPromise;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
