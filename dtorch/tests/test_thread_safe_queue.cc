/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "dtorch/common/thread_safe_queue.h"
#include "test.h"

TEST(ThreadSafeQueueTest, SimpleTest) {
    dtorch::ReaderWriterQueue<int> queue;

    auto ProducerFunc = [&]() {
        queue.Enqueue(1);
        queue.Enqueue(2);
    };

    auto ConsumerFunc = [&]() {
        int element;
        EXPECT_TRUE(queue.TryDequeue(element));
        EXPECT_TRUE(element == 1);
        EXPECT_TRUE(queue.TryDequeue(element));
        EXPECT_TRUE(element == 2);
        EXPECT_FALSE(queue.TryDequeue(element));
    };

    std::thread producerThread(ProducerFunc);
    std::thread consumerThread(ConsumerFunc);
    producerThread.join();
    consumerThread.join();
}
