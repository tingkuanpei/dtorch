/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <string>

#include "dtorch/common/utilities.h"
#include "void_promise_future.h"

namespace dtorch {
namespace core {
namespace communication {

// Forward declaration — defined in file_void_promise_future.cc
struct VoidPromiseFutureShmImpl;

// ============================================================
// class FileVoidFuture — Boost IPC shared memory based Future
// ============================================================
//
// 基于 Boost Interprocess 共享内存实现，用于跨进程传递 void 完成信号。
// 构造函数打开已有共享内存文件，内部通过 VoidPromiseFutureShmImpl 持有持久连接。
// Wait() 在共享内存的 InterprocessCondition 上等待，直到
//   FileVoidPromise::SetValue() 写入 hasValue 标志后唤醒。
// 共享内存中仅存储 hasValue 标志（无需 Tensor 数据）。
// IsReady() 检查 hasValue 标志（非阻塞）。

class FileVoidFuture : public VoidFuture {
public:
    explicit FileVoidFuture(const std::string& shmFileName);

    ~FileVoidFuture() override;

    DTORCH_DISABLE_COPY_AND_MOVE(FileVoidFuture);

    void Get() override;

    void Wait() override;

    bool WaitFor(int64_t timeoutMs) override;

    bool IsReady() const override;

private:
    std::shared_ptr<VoidPromiseFutureShmImpl> mShmImpl;
    bool mValueConsumed;
};

// ============================================================
// class FileVoidPromise — Boost IPC shared memory based Promise
// ============================================================
//
// 基于 Boost Interprocess 共享内存实现，用于跨进程传递 void 完成信号。
// 构造函数（Create 模式）创建新的共享内存文件，初始化 Mutex、Cond、HasValue。
// Deserialize()（Attach 模式）打开已有共享内存文件，用于 Worker 侧反序列化后重建。
// SetValue() 设置 hasValue 标志并通过 InterprocessCondition 唤醒等待的 Future。
// GetFuture() 返回关联的 FileVoidFuture（仅可调用一次，重复调用 DLogFatal）。
// Serialize() 返回共享内存文件名，用于跨进程序列化时传递给 Worker。
// 适用于单机多进程场景（GPU + perDevicePerProcess=true）。

class FileVoidPromise : public VoidPromise {
public:
    // Create mode: creates new shared memory
    FileVoidPromise();

    // Attach mode: used before Deserialize(), does not create shared memory
    // Use Deserialize() to attach to existing shared memory

    ~FileVoidPromise() override;

    DTORCH_DISABLE_COPY_AND_MOVE(FileVoidPromise);

    void SetValue() override;

    std::unique_ptr<VoidFuture> GetFuture() override;

    VoidPromiseType GetType() const override { return VoidPromiseType::kFile; }

    std::string Serialize() const override;

    void Deserialize(const std::string& data) override;

private:
    void CreateSharedMemory();
    void OpenSharedMemory();

private:
    std::string mShmFileName;
    bool mIsCreator;
    bool mFutureTaken;
    std::shared_ptr<VoidPromiseFutureShmImpl> mShmImpl;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
