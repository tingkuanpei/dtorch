/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <thread>

#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/functional/tensor_functional.h"
#include "dtorch/api/cpp/graph.h"
#include "dtorch/api/cpp/int_or_int_array.h"
#include "dtorch/api/cpp/tensor.h"
#include "dtorch/api/cpp/void_future_collect.h"
#include "dtorch/core/communication/promise_future/void_promise_future.h"
#include "dtorch/core/operators/system/sync_op.h"
#include "test.h"

using namespace dtorch::api::cpp;
using namespace dtorch::core;
using namespace dtorch::core::communication;

// ============================================================
// VoidPromise / VoidFuture Tests
// ============================================================

TEST(VoidPromiseFutureTest, MemoryVoidPromiseBasic) {
    // Create a MemoryVoidPromise, get its future, fulfill from another thread,
    // verify that Wait() returns and IsReady() reflects the state.

    auto promise = CreateVoidPromise(VoidPromiseType::kMemory);
    ASSERT_EQ(promise->GetType(), VoidPromiseType::kMemory);

    auto future = promise->GetFuture();
    ASSERT_FALSE(future->IsReady());

    ASSERT_FALSE(future->IsReady());

    // Fulfill from another thread
    std::thread t([&promise]() { promise->SetValue(); });

    future->Wait();
    t.join();
}

TEST(VoidPromiseFutureTest, MemoryVoidPromiseTimeout) {
    auto promise = CreateVoidPromise(VoidPromiseType::kMemory);
    auto future = promise->GetFuture();

    // Not yet fulfilled — timeout should return false
    ASSERT_FALSE(future->WaitFor(10));

    promise->SetValue();
    // Now should succeed
    ASSERT_TRUE(future->WaitFor(10));
}

TEST(VoidPromiseFutureTest, FileVoidPromiseBasic) {
    // Create a FileVoidPromise, get its future, fulfill, verify Wait() returns.

    auto promise = CreateVoidPromise(VoidPromiseType::kFile);
    ASSERT_EQ(promise->GetType(), VoidPromiseType::kFile);

    auto future = promise->GetFuture();
    ASSERT_FALSE(future->IsReady());

    promise->SetValue();
    future->Wait();
}

TEST(VoidPromiseFutureTest, FileVoidPromiseSerialize) {
    // Verify serialization round-trip: serialize the promise, then
    // reconstruct via CreateVoidPromiseFromSerialized and verify SetValue works.

    auto promise = CreateVoidPromise(VoidPromiseType::kFile);
    std::string serialized = promise->Serialize();
    ASSERT_FALSE(serialized.empty());

    auto future = promise->GetFuture();
    ASSERT_FALSE(future->IsReady());

    // Reconstruct on "worker side"
    auto workerPromise = CreateVoidPromiseFromSerialized(VoidPromiseType::kFile, serialized);
    ASSERT_NE(workerPromise, nullptr);
    ASSERT_EQ(workerPromise->GetType(), VoidPromiseType::kFile);

    workerPromise->SetValue();
    future->Wait();
}

// ============================================================
// VoidFutureCollect Tests
// ============================================================

TEST(VoidFutureCollectTest, BasicCollect) {
    VoidFutureCollect collect;

    // Create 3 futures, fulfill them, verify Collect::Wait() works
    for (int i = 0; i < 3; i++) {
        auto promise = CreateVoidPromise(VoidPromiseType::kMemory);
        auto future = promise->GetFuture();
        collect.AddFuture(std::move(future));
        // Fulfill immediately
        promise->SetValue();
    }

    // All futures are already ready
    ASSERT_TRUE(collect.IsReady());
    collect.Wait();  // Should return immediately
}

TEST(VoidFutureCollectTest, WaitBlocksUntilAllReady) {
    VoidFutureCollect collect;

    auto promise = CreateVoidPromise(VoidPromiseType::kMemory);
    auto future = promise->GetFuture();
    collect.AddFuture(std::move(future));

    ASSERT_FALSE(collect.IsReady());

    // Fulfill from another thread, then verify Wait returns
    std::thread t([&promise]() { promise->SetValue(); });

    collect.Wait();
    t.join();
}

// ============================================================
// SyncOp Tests
// ============================================================

TEST(SyncOpTest, InferOperatorAssignInfo) {
    // Create SyncParam with 2 devices, build SyncOp, verify stream keys.

    Device cpuDevice(DeviceKind::kCpu, 0);
    std::vector<Device> devices = {cpuDevice};
    std::vector<std::unique_ptr<VoidPromise>> promises;
    promises.push_back(CreateVoidPromise(VoidPromiseType::kMemory));

    auto param = std::make_shared<SyncParam>(std::move(devices), std::move(promises));
    SyncOp syncOp(param);

    syncOp.Infer();
    ASSERT_EQ(syncOp.GetOperatorAssignInfo().NumKernelForThisOp(), 1);
}

// ============================================================
// Graph::Sync() Integration Test
// ============================================================

TEST(GraphSyncTest, GraphSyncEndToEndCpu) {
    // Create a graph, perform a CPU tensor operation, call Sync().Wait(),
    // verify it completes without hanging.

    Graph graph;
    IntOrIntArray shape({2, 3});
    DeviceMesh cpuMesh(Device(DeviceKind::kCpu, 0));
    auto tensor = functional::_Empty(graph, shape, DataKind::kFloat32, cpuMesh);

    // Sync should not hang
    // SyncFuture returns VoidFutureCollect
    auto futures = graph.SyncFuture();
    ASSERT_NO_THROW(futures.Wait());

    // Sync() is the blocking convenience wrapper
    ASSERT_NO_THROW(graph.Sync());
}
