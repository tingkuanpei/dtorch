/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "eager_graph_executor_message.h"

#include "dtorch/core/graph/eager_graph_executor_message_imp.h"
#include "dtorch/external/python/python_gil.h"

namespace dtorch {
namespace core {

EGEMessageQueue::EGEMessageQueue() : mGetDestroySignal(false), mMessageQueue() {}

void EGEMessageQueue::Destroy() {
    DDebugAssert(!mGetDestroySignal);
    mGetDestroySignal = true;
    mMessageQueue.Enqueue(nullptr);
}

void EGEMessageQueue::ProcessMessages(size_t requireSize, EagerGraphExecutor& eagerGraphExecutor) {
    DDebugAssert(requireSize > 0);
    size_t getMsgCount = 0;
    while (getMsgCount < requireSize) {
        std::unique_ptr<EGEMessage> message;
        bool dequeueSuccess = mMessageQueue.TryDequeue(message);
        if (!dequeueSuccess && getMsgCount == 0) {
            DAlwaysAssert(message == nullptr);
            mMessageQueue.WaitDequeue(message);
            dequeueSuccess = true;
        }

        if (mGetDestroySignal) {
            return;
        }

        if (!dequeueSuccess) {
            return;
        }

        getMsgCount++;
        DAlwaysAssert(message != nullptr);
        message->ProcessEGEMessage(eagerGraphExecutor);
    }
}

void EGEMessageQueue::PushMessageImp(std::unique_ptr<EGEMessage> message) {
    DDebugAssert(message != nullptr);
    mMessageQueue.Enqueue(std::move(message));
}

void EGEMessageQueue::PushMessage(std::unique_ptr<EGEMessage> message) {
    auto scopedRelease = external::python::GetPythonGilScopedRelease();

    DDebugAssert(!message->BlockProducerThread());
    PushMessageImp(std::move(message));
}

template <typename ResultType>
ResultType EGEMessageQueue::PushMessageAndGetResult(std::unique_ptr<FutureEGEMessage<ResultType>> message) {
    auto scopedRelease = external::python::GetPythonGilScopedRelease();

    DDebugAssert(message->BlockProducerThread());
    std::future<ResultType> future = message->GetFuture();
    PushMessageImp(std::move(message));
    return future.get();
}

template void EGEMessageQueue::PushMessageAndGetResult<void>(std::unique_ptr<FutureEGEMessage<void>> message);

}  // namespace core
}  // namespace dtorch
