/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <algorithm>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>

#include "dtorch/common/thread_safe_queue.h"
#include "dtorch/core/operators/operator.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

class EagerGraphExecutor;

// EagerGraphExecutorMessage
class EGEMessage {
public:
    EGEMessage() = default;

    virtual ~EGEMessage() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(EGEMessage);

    virtual bool BlockProducerThread() { return false; }

    virtual void ProcessEGEMessage(EagerGraphExecutor& eagerGraphExecutor) = 0;
};

// EagerGraphExecutorMessage with future as return value
template <typename ResultType>
class FutureEGEMessage : public EGEMessage {
public:
    FutureEGEMessage() : mPromise() {}

    std::future<ResultType> GetFuture() { return mPromise.get_future(); }

    bool BlockProducerThread() override { return true; }

protected:
    std::promise<ResultType> mPromise;
};

class EGEMessageQueue {
public:
    EGEMessageQueue();

    // Call by producer thread
    void Destroy();

    // Call by producer thread
    void PushMessage(std::unique_ptr<EGEMessage> message);

    // Call by producer thread
    template <typename ResultType>
    ResultType PushMessageAndGetResult(std::unique_ptr<FutureEGEMessage<ResultType>> message);

    // Call by consumer thread
    void ProcessMessages(size_t requireSize, EagerGraphExecutor& eagerGraphExecutor);

private:
    void PushMessageImp(std::unique_ptr<EGEMessage> message);

private:
    std::atomic_bool mGetDestroySignal;
    BlockingReaderWriterQueue<std::unique_ptr<EGEMessage>, 2048> mMessageQueue;
};

}  // namespace core
}  // namespace dtorch
