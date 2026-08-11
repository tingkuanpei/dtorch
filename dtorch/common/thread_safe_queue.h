/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif
#include <concurrentqueue.h>
#include <readerwriterqueue.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "utilities.h"

namespace dtorch {

// https://github.com/cameron314/readerwriterqueue

template <typename T, size_t MAX_BLOCK_SIZE = 512>
class ReaderWriterQueue {
public:
    explicit ReaderWriterQueue() : mQueue(64) {}

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(ReaderWriterQueue);

    DTORCH_FORCEINLINE bool Enqueue(const T& element) { return mQueue.enqueue(element); }

    DTORCH_FORCEINLINE bool Enqueue(T&& element) { return mQueue.enqueue(std::forward<T>(element)); }

    template <typename U>
    DTORCH_FORCEINLINE bool TryDequeue(U& element) {
        return mQueue.try_dequeue(element);
    }

    size_t SizeApprox() const { return mQueue.size_approx(); }

private:
    moodycamel::ReaderWriterQueue<T, MAX_BLOCK_SIZE> mQueue;
};

template <typename T, size_t MAX_BLOCK_SIZE = 512>
class BlockingReaderWriterQueue {
public:
    explicit BlockingReaderWriterQueue() : mQueue(64) {}

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(BlockingReaderWriterQueue);

    DTORCH_FORCEINLINE bool Enqueue(const T& element) { return mQueue.enqueue(element); }

    DTORCH_FORCEINLINE bool Enqueue(T&& element) { return mQueue.enqueue(std::forward<T>(element)); }

    template <typename U>
    DTORCH_FORCEINLINE bool TryDequeue(U& element) {
        return mQueue.try_dequeue(element);
    }

    template <typename U>
    DTORCH_FORCEINLINE void WaitDequeue(U& element) {
        mQueue.wait_dequeue(element);
    }

    template <typename U>
    DTORCH_FORCEINLINE void WaitDequeueTimed(U& element, std::int64_t timeout_usecs) {
        mQueue.wait_dequeue_timed(element, timeout_usecs);
    }

    size_t SizeApprox() const { return mQueue.size_approx(); }

private:
    moodycamel::BlockingReaderWriterQueue<T, MAX_BLOCK_SIZE> mQueue;
};

// https://github.com/cameron314/concurrentqueue

template <typename T>
class ConcurrentQueue {
public:
    explicit ConcurrentQueue() : mQueue() {}

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(ConcurrentQueue);

    DTORCH_FORCEINLINE bool Enqueue(const T& element) { return mQueue.enqueue(element); }

    DTORCH_FORCEINLINE bool Enqueue(T&& element) { return mQueue.enqueue(std::forward<T>(element)); }

    template <typename U>
    DTORCH_FORCEINLINE bool TryDequeue(U& value) {
        return mQueue.try_dequeue(value);
    }

    size_t SizeApprox() const { return mQueue.size_approx(); }

private:
    moodycamel::ConcurrentQueue<T> mQueue;
};

}  // namespace dtorch
