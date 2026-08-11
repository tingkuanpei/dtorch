/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "dtorch/common/thread_safe_queue.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/kernel/kernel.h"

namespace dtorch {
namespace core {

class KernelStream {
public:
    KernelStream(const Device& device, KernelStreamType streamType, bool isAsync);

    virtual ~KernelStream() { Destroy(); }

    DTORCH_DISABLE_COPY_AND_MOVE(KernelStream);

    void LaunchKernel(std::unique_ptr<Kernel> kernel);

    // Same as cudaStreamSynchronize: wait for all kernel launch and finish
    // Important: not call inside Kernel::Run(), because it will cause deadlock.
    void Sync();

    DTORCH_FORCEINLINE const Device& GetDevice() const noexcept { return mDevice; }

    DTORCH_FORCEINLINE DeviceKind GetDeviceKind() const noexcept { return mDevice.deviceKind; }

    DTORCH_FORCEINLINE KernelStreamType GetStreamType() const noexcept { return mStreamType; }

    DTORCH_FORCEINLINE std::thread::id GetAsyncThreadId() const noexcept { return mAsyncStreamThreadId; }

protected:
    void Init();

    virtual void InitInAsyncThread() {}

    virtual void SyncImp();

    void AsyncMain();

private:
    void Destroy();

    void SendDestroyMessage();

protected:
    struct SharedInfo {
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic_bool destroy;
        std::atomic_bool sync;

    public:
        SharedInfo() : mutex(), cv(), destroy(false), sync(false) {}
    };
    using KernelQueue = BlockingReaderWriterQueue<std::unique_ptr<Kernel>>;

    // Local Device
    Device mDevice;
    KernelStreamType mStreamType;
    bool mIsAsync;
    std::thread mAsyncStreamThread;
    std::thread::id mAsyncStreamThreadId;
    KernelQueue mKernelQueue;
    SharedInfo mSharedInfo;
};

}  // namespace core
}  // namespace dtorch
