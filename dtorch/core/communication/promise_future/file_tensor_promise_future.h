/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <string>

#include "dtorch/common/utilities.h"
#include "tensor_promise_future.h"

namespace dtorch {
namespace core {
namespace communication {

// Forward declaration — defined in file_tensor_promise_future.cc
struct FileTensorPromiseFutureShmImpl;

// ============================================================
// class FileTensorFuture — Boost IPC shared memory based Future
// ============================================================
//
// 基于 Boost Interprocess 共享内存实现，用于跨进程传递 Tensor 值。
// 构造函数打开已有共享内存文件，内部通过 FileTensorPromiseFutureShmImpl 持有持久连接。
// Get() / Wait() 在共享内存的 InterprocessCondition 上等待，直到
//   FileTensorPromise::SetValue() 写入 hasValue 标志和 Tensor 数据后唤醒。
// 共享内存中存储的是 TorchUtil::ToIpcMemHandle() 序列化后的 tensor 数据。
// IsReady() 检查 hasValue 标志（非阻塞）。

class FileTensorFuture : public TensorFuture {
public:
    explicit FileTensorFuture(const std::string& shmFileName);

    ~FileTensorFuture() override;

    DTORCH_DISABLE_COPY_AND_MOVE(FileTensorFuture);

    std::shared_ptr<torch::Tensor> Get() override;

    void Wait() override;

    bool WaitFor(int64_t timeoutMs) override;

    bool IsReady() const override;

private:
    std::shared_ptr<FileTensorPromiseFutureShmImpl> mShmImpl;
    bool mValueConsumed;
};

// ============================================================
// class FileTensorPromise — Boost IPC shared memory based Promise
// ============================================================
//
// 基于 Boost Interprocess 共享内存实现，用于跨进程传递 Tensor 值。
// 构造函数（Create 模式）创建新的共享内存文件，初始化 Mutex、Cond、HasValue。
// Deserialize()（Attach 模式）打开已有共享内存文件，用于 Worker 侧反序列化后重建。
// SetValue() 将 tensor 通过 TorchUtil::ToIpcMemHandle() 序列化后写入共享内存，
//   设置 hasValue 标志并通过 InterprocessCondition 唤醒等待的 Future。
// GetFuture() 返回关联的 FileTensorFuture（仅可调用一次，重复调用 DLogFatal）。
// Serialize() 返回共享内存文件名，用于跨进程序列化时传递给 Worker。
// 适用于单机多进程场景（GPU + perDevicePerProcess=true）。

class FileTensorPromise : public TensorPromise {
public:
    // Create mode: creates new shared memory
    FileTensorPromise();

    // Attach mode: used before Deserialize(), does not create shared memory
    // Use Deserialize() to attach to existing shared memory

    ~FileTensorPromise() override;

    DTORCH_DISABLE_COPY_AND_MOVE(FileTensorPromise);

    void SetValue(std::shared_ptr<torch::Tensor> tensor) override;

    std::unique_ptr<TensorFuture> GetFuture() override;

    TensorPromiseType GetType() const override { return TensorPromiseType::kFile; }

    std::string Serialize() const override;

    void Deserialize(const std::string& data) override;

private:
    void CreateSharedMemory();
    void OpenSharedMemory();

private:
    std::string mShmFileName;
    bool mIsCreator;
    bool mFutureTaken;
    std::shared_ptr<FileTensorPromiseFutureShmImpl> mShmImpl;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
