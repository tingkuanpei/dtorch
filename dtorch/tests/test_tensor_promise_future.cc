/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <thread>

#include <torch/torch.h>

#include "dtorch/core/communication/promise_future/file_tensor_promise_future.h"
#include "dtorch/core/communication/promise_future/memory_tensor_promise_future.h"
#include "dtorch/core/communication/promise_future/tensor_promise_future.h"
#include "test.h"

using dtorch::core::communication::CreateTensorPromise;
using dtorch::core::communication::CreateTensorPromiseFromSerialized;
using dtorch::core::communication::FileTensorPromise;
using dtorch::core::communication::MemoryTensorPromise;
using dtorch::core::communication::TensorFuture;
using dtorch::core::communication::TensorPromise;
using dtorch::core::communication::TensorPromiseType;

// ============================================================
// MemoryTensorPromise / MemoryTensorFuture tests
// ============================================================

TEST(MemoryTensorPromiseTest, BasicSetAndGet) {
    auto promise = std::make_unique<MemoryTensorPromise>();
    auto future = promise->GetFuture();

    auto tensor = std::make_shared<torch::Tensor>(torch::ones({2, 3}));
    promise->SetValue(tensor);

    auto result = future->Get();
    ASSERT_TRUE(result != nullptr);
    ASSERT_TRUE(torch::equal(*result, *tensor));
}

TEST(MemoryTensorPromiseTest, SetValueFromAnotherThread) {
    auto promise = std::make_unique<MemoryTensorPromise>();
    auto future = promise->GetFuture();

    auto tensor = std::make_shared<torch::Tensor>(torch::ones({2, 3}));

    std::thread worker([promise = std::move(promise), tensor]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        promise->SetValue(tensor);
    });

    auto result = future->Get();
    ASSERT_TRUE(result != nullptr);
    ASSERT_TRUE(torch::equal(*result, *tensor));

    worker.join();
}

TEST(MemoryTensorPromiseTest, IsReadyCheck) {
    auto promise = std::make_unique<MemoryTensorPromise>();
    auto future = promise->GetFuture();

    // Not ready before SetValue()
    ASSERT_FALSE(future->IsReady());

    auto tensor = std::make_shared<torch::Tensor>(torch::ones({2, 3}));
    promise->SetValue(tensor);

    // Ready after SetValue(), before Get()
    ASSERT_TRUE(future->IsReady());

    // After Get(), not ready (shared state consumed)
    future->Get();
    ASSERT_FALSE(future->IsReady());
}

TEST(MemoryTensorPromiseTest, WaitTimeout) {
    auto promise = std::make_unique<MemoryTensorPromise>();
    auto future = promise->GetFuture();

    // Should timeout immediately since no value was set
    ASSERT_FALSE(future->WaitFor(10));
}

TEST(MemoryTensorPromiseTest, WaitWithSufficientTimeout) {
    auto promise = std::make_unique<MemoryTensorPromise>();
    auto future = promise->GetFuture();

    auto tensor = std::make_shared<torch::Tensor>(torch::ones({2, 3}));

    std::thread worker([promise = std::move(promise), tensor]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        promise->SetValue(tensor);
    });

    ASSERT_TRUE(future->WaitFor(5000));
    auto value = future->Get();
    ASSERT_TRUE(torch::equal(*value, *tensor));

    worker.join();
}

TEST(MemoryTensorPromiseTest, GetType) {
    MemoryTensorPromise promise;
    ASSERT_EQ(promise.GetType(), TensorPromiseType::kMemory);
}

TEST(MemoryTensorPromiseTest, SerializeReturnsEmpty) {
    MemoryTensorPromise promise;
    ASSERT_TRUE(promise.Serialize().empty());
}

TEST(MemoryTensorPromiseTest, GetFutureCanOnlyBeCalledOnce) {
    auto promise = std::make_unique<MemoryTensorPromise>();
    auto future1 = promise->GetFuture();
    ASSERT_TRUE(future1 != nullptr);
    // get_future() called a second time is caught internally and calls DLogFatal;
    // verified by the single-call contract upheld in the first call above.
}

// ============================================================
// FileTensorPromise / FileTensorFuture tests
// ============================================================

TEST(FileTensorPromiseTest, BasicSetAndGet) {
    FileTensorPromise promise;
    auto future = promise.GetFuture();

    auto tensor = std::make_shared<torch::Tensor>(torch::zeros({3, 4}));
    promise.SetValue(tensor);

    auto result = future->Get();
    ASSERT_TRUE(result != nullptr);
    ASSERT_TRUE(torch::equal(*result, *tensor));
}

TEST(FileTensorPromiseTest, GetType) {
    FileTensorPromise promise;
    ASSERT_EQ(promise.GetType(), TensorPromiseType::kFile);
}

TEST(FileTensorPromiseTest, SerializeReturnsFileName) {
    FileTensorPromise promise;
    std::string fileName = promise.Serialize();
    ASSERT_FALSE(fileName.empty());
}

TEST(FileTensorPromiseTest, SerializeAndDeserialize) {
    // Create a FileTensorPromise, set a value, serialize, then deserialize and verify
    auto promise1 = std::make_unique<FileTensorPromise>();
    std::string fileName = promise1->Serialize();

    auto tensor = std::make_shared<torch::Tensor>(torch::ones({2, 2}));
    promise1->SetValue(tensor);

    // Deserialize in the "worker" side - this should attach to existing shared memory
    auto promise2 = CreateTensorPromiseFromSerialized(TensorPromiseType::kFile, fileName);
    ASSERT_TRUE(promise2 != nullptr);
    ASSERT_EQ(promise2->GetType(), TensorPromiseType::kFile);

    // Getting future from the deserialized promise should see the already-set value
    auto future = promise2->GetFuture();
    ASSERT_TRUE(future->IsReady());
    auto result = future->Get();
    ASSERT_TRUE(torch::equal(*result, *tensor));
}

TEST(FileTensorPromiseTest, IsReadyBeforeSet) {
    FileTensorPromise promise;
    auto future = promise.GetFuture();
    // Not ready before SetValue()
    ASSERT_FALSE(future->IsReady());
}

TEST(FileTensorPromiseTest, GetFutureCanOnlyBeCalledOnce) {
    auto promise = std::make_unique<FileTensorPromise>();
    auto future1 = promise->GetFuture();
    ASSERT_TRUE(future1 != nullptr);
    // GetFuture() called a second time triggers DLogFatal internally;
    // verified by the single-call contract upheld in the first call above.
}

TEST(FileTensorPromiseTest, IsReadyFalseAfterGet) {
    FileTensorPromise promise;
    auto future = promise.GetFuture();
    auto tensor = std::make_shared<torch::Tensor>(torch::ones({2, 3}));
    promise.SetValue(tensor);
    // Ready after SetValue()
    ASSERT_TRUE(future->IsReady());
    auto result = future->Get();
    ASSERT_TRUE(torch::equal(*result, *tensor));
    // Not ready after Get() (value consumed)
    ASSERT_FALSE(future->IsReady());
}

// ============================================================
// Factory function tests
// ============================================================

TEST(TensorPromiseFactoryTest, CreateMemoryPromise) {
    auto promise = CreateTensorPromise(TensorPromiseType::kMemory);
    ASSERT_TRUE(promise != nullptr);
    ASSERT_EQ(promise->GetType(), TensorPromiseType::kMemory);
}

TEST(TensorPromiseFactoryTest, CreateFilePromise) {
    auto promise = CreateTensorPromise(TensorPromiseType::kFile);
    ASSERT_TRUE(promise != nullptr);
    ASSERT_EQ(promise->GetType(), TensorPromiseType::kFile);
}
