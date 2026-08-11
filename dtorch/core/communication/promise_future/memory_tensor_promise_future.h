/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <future>
#include <memory>

#include "dtorch/common/debug.h"
#include "tensor_promise_future.h"

namespace dtorch {
namespace core {
namespace communication {

// ============================================================
// class MemoryTensorFuture — std::future based Future
// ============================================================
//
// 基于 std::future<std::shared_ptr<torch::Tensor>> 实现。
// 内部持有一个 std::future，Get() 调用 std::future::get() 阻塞等待，
// IsReady() 通过 wait_for(0ms) 非阻塞检查 ready 状态。
// 适用于单机多线程场景（同进程内传递），零额外开销。

class MemoryTensorFuture : public TensorFuture {
public:
    explicit MemoryTensorFuture(std::future<std::shared_ptr<torch::Tensor>> future);

    ~MemoryTensorFuture() override = default;

    DTORCH_DISABLE_COPY_AND_MOVE(MemoryTensorFuture);

    std::shared_ptr<torch::Tensor> Get() override;

    void Wait() override;

    bool WaitFor(int64_t timeoutMs) override;

    bool IsReady() const override;

private:
    std::future<std::shared_ptr<torch::Tensor>> mFuture;
};

// ============================================================
// class MemoryTensorPromise — std::promise based Promise
// ============================================================
//
// 基于 std::promise<std::shared_ptr<torch::Tensor>> 实现。
// SetValue() 调用 std::promise::set_value() 写入结果，
// GetFuture() 调用 std::promise::get_future() 创建关联的 MemoryTensorFuture
//   （仅可调用一次，重复调用时捕获 std::future_error 后 DLogFatal）。
// Serialize() / Deserialize() 不支持（同进程内无需序列化）。
// 适用于单机多线程场景，零额外开销。

class MemoryTensorPromise : public TensorPromise {
public:
    MemoryTensorPromise();

    ~MemoryTensorPromise() override = default;

    DTORCH_DISABLE_COPY_AND_MOVE(MemoryTensorPromise);

    void SetValue(std::shared_ptr<torch::Tensor> tensor) override;

    std::unique_ptr<TensorFuture> GetFuture() override;

    TensorPromiseType GetType() const override { return TensorPromiseType::kMemory; }

    std::string Serialize() const override { return ""; }

    void Deserialize(const std::string& data) override {
        IgnoreUnused(data);
        DDebugAssert(false && "MemoryTensorPromise does not support deserialization");
    }

private:
    std::promise<std::shared_ptr<torch::Tensor>> mPromise;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
