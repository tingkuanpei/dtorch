/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "kernel_stream.h"

#include <pthread.h>
#include <sched.h>

#include "dtorch/common/debug.h"
#include "dtorch/core/communication/tensor_store/tensor_store.h"

namespace dtorch {
namespace core {

namespace {

void SetAsyncStreamThreadHighestPriority(pthread_t handle) noexcept {
    struct sched_param param = {};
    const int max_prio = sched_get_priority_max(SCHED_FIFO);
    if (max_prio < 0) {
        return;
    }
    param.sched_priority = max_prio;
    (void)pthread_setschedparam(handle, SCHED_FIFO, &param);
}

}  // namespace

KernelStream::KernelStream(const Device& device, KernelStreamType streamType, bool isAsync)
    : mDevice(device),
      mStreamType(streamType),
      mIsAsync(isAsync),
      mAsyncStreamThread(),
      mAsyncStreamThreadId(),
      mKernelQueue(),
      mSharedInfo() {}

void KernelStream::LaunchKernel(std::unique_ptr<Kernel> kernel) {
    DDebugAssert(mIsAsync);
    DDebugAssert(mAsyncStreamThread.joinable());
    DAlwaysAssert(mKernelQueue.Enqueue(std::move(kernel)));
}

void KernelStream::Sync() {
    {
        std::unique_lock<std::mutex> lock(mSharedInfo.mutex);
        // Support multiple threads calling Sync concurrently
        while (mSharedInfo.sync) {
            mSharedInfo.cv.wait(lock, [&] { return !mSharedInfo.sync; });
            if (!mSharedInfo.sync) {
                break;
            }
        }

        DAlwaysAssert(!mSharedInfo.sync);
        mSharedInfo.sync = true;
        LaunchKernel(nullptr);
        mSharedInfo.cv.wait(lock, [&] { return !mSharedInfo.sync; });
    }
}

void KernelStream::SyncImp() {
    std::unique_lock<std::mutex> lock(mSharedInfo.mutex);
    mSharedInfo.sync = false;
    lock.unlock();
    mSharedInfo.cv.notify_all();
}

void KernelStream::AsyncMain() {
    InitInAsyncThread();

    while (true) {
        std::unique_ptr<Kernel> kernels;
        mKernelQueue.WaitDequeue(kernels);

        if (mSharedInfo.destroy) {
            break;
        }

        if (kernels == nullptr) {
            DAlwaysAssert(mSharedInfo.sync);
            SyncImp();
        } else {
            // After the kernel execution, release the kernel immediately to free up the input and output tensors of
            // the kernel at the earliest opportunity, thereby reducing max memory occupation.
            kernels->Run();
            kernels.reset();
        }
    }
}

void KernelStream::Init() {
    if (mIsAsync) {
        mAsyncStreamThread = std::thread(&KernelStream::AsyncMain, this);
        mAsyncStreamThreadId = mAsyncStreamThread.get_id();
        SetAsyncStreamThreadHighestPriority(mAsyncStreamThread.native_handle());
    } else {
        InitInAsyncThread();
        mAsyncStreamThreadId = std::this_thread::get_id();
    }
}

void KernelStream::Destroy() {
    if (mAsyncStreamThread.joinable()) {
        SendDestroyMessage();
        mAsyncStreamThread.join();
    }
}

void KernelStream::SendDestroyMessage() {
    mSharedInfo.destroy = true;
    LaunchKernel(nullptr);
}

}  // namespace core
}  // namespace dtorch
