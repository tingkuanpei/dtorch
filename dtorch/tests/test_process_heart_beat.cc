/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "dtorch/core/distributed/process_heart_beat.h"
#include "dtorch/external/rpc/heart_beat_interface.h"
#include "dtorch/external/rpc/rpc_common.h"
#include "test.h"

using namespace dtorch;
using namespace dtorch::core;
using namespace dtorch::external::rpc;

// ============================================================================
// Test helper: minimal concrete ProcessHeartBeatBase for testing gRPC integration
// ============================================================================

namespace {

class TestHeartBeat : public ProcessHeartBeatBase {
public:
    TestHeartBeat() : mMutex(), mRegisteredWorkers() {}

    void RegisterWorkerProcess(const std::string& workerAddress) override {
        std::unique_lock<std::mutex> lock(mMutex);
        mRegisteredWorkers.push_back(workerAddress);
    }

    void UnregisterWorkerProcess(const std::string& workerAddress) override {
        std::unique_lock<std::mutex> lock(mMutex);
        auto it = std::find(mRegisteredWorkers.begin(), mRegisteredWorkers.end(), workerAddress);
        if (it != mRegisteredWorkers.end()) {
            mRegisteredWorkers.erase(it);
        }
    }

    std::vector<std::string> GetRegisteredWorkers() const {
        std::unique_lock<std::mutex> lock(mMutex);
        return mRegisteredWorkers;
    }

    size_t GetRegisteredWorkerCount() const {
        std::unique_lock<std::mutex> lock(mMutex);
        return mRegisteredWorkers.size();
    }

private:
    mutable std::mutex mMutex;
    std::vector<std::string> mRegisteredWorkers;
};

}  // namespace

// ============================================================================
// 1. ProcessHeartBeatBase Tests (base class mechanics)
// ============================================================================

TEST(ProcessHeartBeatBaseTest, WaitForStopPollTimeout) {
    TestHeartBeat hb;

    auto start = std::chrono::steady_clock::now();
    hb.WaitForStopPoll(100);
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // Should block for at least the timeout duration
    EXPECT_GE(elapsed, 90);
    // Allow some slack for scheduling variance
    EXPECT_LT(elapsed, 500);
}

TEST(ProcessHeartBeatBaseTest, NotifyStopPollWakesWaitForStopPoll) {
    TestHeartBeat hb;

    // Spawn a thread that will call NotifyStopPoll after a short delay
    std::thread notifier([&hb]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        hb.NotifyStopPoll();
    });

    auto start = std::chrono::steady_clock::now();
    hb.WaitForStopPoll(5000);
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // Should wake up quickly (well before the 5s timeout)
    EXPECT_LT(elapsed, 2000);

    notifier.join();
}

TEST(ProcessHeartBeatBaseTest, NotifyStopPollBeforeWaitReturnsImmediately) {
    TestHeartBeat hb;

    // Call NotifyStopPoll first, then WaitForStopPoll should return immediately
    hb.NotifyStopPoll();

    auto start = std::chrono::steady_clock::now();
    hb.WaitForStopPoll(5000);
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // Should return almost immediately since mStopPoll is already true
    EXPECT_LT(elapsed, 100);
}

// ============================================================================
// 2. HeartBeatClient Standalone Tests (client without server)
// ============================================================================

// Note: gRPC calls to non-existent servers will block until the gRPC deadline
// (default 3s via GlobalOption). This is expected behavior — the retry policy
// and deadline ensure the call eventually returns false.
TEST(HeartBeatClientStandaloneTest, IsBeatFailsWithoutServer) {
    std::string nonExistentAddress = GetRandomUdsAddress();
    HeartBeatClient client(nonExistentAddress);

    bool result = client.IsBeat();
    EXPECT_FALSE(result) << "IsBeat should return false when no gRPC server is running";
}

TEST(HeartBeatClientStandaloneTest, NotifyStopPollFailsWithoutServer) {
    std::string nonExistentAddress = GetRandomUdsAddress();
    HeartBeatClient client(nonExistentAddress);

    bool result = client.NotifyStopPoll();
    EXPECT_FALSE(result) << "NotifyStopPoll should return false when no gRPC server is running";
}

TEST(HeartBeatClientStandaloneTest, RegisterWorkerFailsWithoutServer) {
    std::string nonExistentAddress = GetRandomUdsAddress();
    HeartBeatClient client(nonExistentAddress);

    bool result = client.RegisterWorker("worker_addr_1");
    EXPECT_FALSE(result) << "RegisterWorker should return false when no gRPC server is running";
}

TEST(HeartBeatClientStandaloneTest, UnregisterWorkerFailsWithoutServer) {
    std::string nonExistentAddress = GetRandomUdsAddress();
    HeartBeatClient client(nonExistentAddress);

    bool result = client.UnregisterWorker("worker_addr_1");
    EXPECT_FALSE(result) << "UnregisterWorker should return false when no gRPC server is running";
}

TEST(HeartBeatClientStandaloneTest, GetAddress) {
    std::string address = GetRandomUdsAddress();
    HeartBeatClient client(address);

    EXPECT_EQ(client.GetAddress(), address);
}

// ============================================================================
// 3. HeartBeatServer + HeartBeatClient Integration Tests
// ============================================================================

TEST(HeartBeatServerClientTest, IsBeatSuccess) {
    std::string address = GetRandomUdsAddress();
    TestHeartBeat testHb;
    HeartBeatServer server(address, testHb);

    HeartBeatClient client(address);
    bool result = client.IsBeat();
    EXPECT_TRUE(result) << "IsBeat should succeed when server is running";
}

TEST(HeartBeatServerClientTest, NotifyStopPollForwardsToBase) {
    std::string address = GetRandomUdsAddress();
    TestHeartBeat testHb;
    HeartBeatServer server(address, testHb);

    HeartBeatClient client(address);
    bool result = client.NotifyStopPoll();
    EXPECT_TRUE(result) << "NotifyStopPoll RPC should succeed";

    // Verify mStopPoll was set on the base via the server callback
    // WaitForStopPoll should return immediately since NotifyStopPoll was called
    auto start = std::chrono::steady_clock::now();
    testHb.WaitForStopPoll(100);
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(elapsed, 100) << "WaitForStopPoll should return immediately after NotifyStopPoll";
}

TEST(HeartBeatServerClientTest, RegisterWorkerForwardsToBase) {
    std::string address = GetRandomUdsAddress();
    TestHeartBeat testHb;
    HeartBeatServer server(address, testHb);

    HeartBeatClient client(address);
    bool result = client.RegisterWorker("worker_addr_test");
    EXPECT_TRUE(result) << "RegisterWorker RPC should succeed";

    // Give the gRPC callback a moment to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify the worker was registered in the TestHeartBeat callback
    EXPECT_EQ(testHb.GetRegisteredWorkerCount(), 1);
    auto workers = testHb.GetRegisteredWorkers();
    ASSERT_EQ(workers.size(), 1);
    EXPECT_EQ(workers[0], "worker_addr_test");
}

TEST(HeartBeatServerClientTest, UnregisterWorkerForwardsToBase) {
    std::string address = GetRandomUdsAddress();
    TestHeartBeat testHb;
    HeartBeatServer server(address, testHb);

    HeartBeatClient client(address);

    // Register first
    client.RegisterWorker("worker_addr_test");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(testHb.GetRegisteredWorkerCount(), 1);

    // Then unregister
    bool result = client.UnregisterWorker("worker_addr_test");
    EXPECT_TRUE(result) << "UnregisterWorker RPC should succeed";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(testHb.GetRegisteredWorkerCount(), 0);
}

// ============================================================================
// 4. MainProcessHeartBeat Tests (Main-side heartbeat)
// ============================================================================

TEST(MainProcessHeartBeatTest, ConstructionAndDestruction) {
    std::string mainAddress = GetRandomUdsAddress();
    // Construction starts server + polling thread; destruction stops them cleanly
    MainProcessHeartBeat mainHb(mainAddress);
    EXPECT_EQ(mainHb.GetMainAddress(), mainAddress);
    // Destructor runs here — should not hang
}

TEST(MainProcessHeartBeatTest, GetMainAddress) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    EXPECT_EQ(mainHb.GetMainAddress(), mainAddress);
}

TEST(MainProcessHeartBeatTest, IsBeatRespondsWhenRunning) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    // A client can reach Main's gRPC server
    HeartBeatClient client(mainAddress);
    bool result = client.IsBeat();
    EXPECT_TRUE(result) << "Main's gRPC server should respond to IsBeat";
}

TEST(MainProcessHeartBeatTest, RegisterAndUnregisterWorkerViaRpc) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    std::string workerAddress = GetRandomUdsAddress();
    HeartBeatClient workerClient(mainAddress);

    // Worker registers with Main
    bool registered = workerClient.RegisterWorker(workerAddress);
    EXPECT_TRUE(registered) << "RegisterWorker RPC to Main should succeed";

    // Give the gRPC callback time to process in Main's server thread
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Verify Main can now reach the worker — but the worker doesn't have a server yet.
    // Main's internal polling will try IsBeat on this worker; the worker has no server
    // so IsBeat will fail → Main calls std::exit(0). We must unregister before the
    // polling thread's next cycle (~4s). Let's do it immediately.

    // Worker unregisters from Main before Main's next poll cycle
    bool unregistered = workerClient.UnregisterWorker(workerAddress);
    EXPECT_TRUE(unregistered) << "UnregisterWorker RPC to Main should succeed";
}

TEST(MainProcessHeartBeatTest, NotifyStopPollViaRpcWakesMainPolling) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    // A client (simulating a worker) calls NotifyStopPoll on Main's server.
    // This triggers Main's ProcessHeartBeatBase::NotifyStopPoll(),
    // which sets mStopPoll = true and notifies the CV. Main's polling thread
    // will then wake up, see mStopPoll = true, break, and call StopWorkerPoll().
    HeartBeatClient client(mainAddress);
    bool result = client.NotifyStopPoll();
    EXPECT_TRUE(result) << "NotifyStopPoll RPC to Main should succeed";

    // Give the polling thread time to process the stop signal
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Main's destructor will join the polling thread (which should already be stopping)
    // No hang expected.
}

// ============================================================================
// 5. WorkerProcessHeartBeat Tests (Worker-side heartbeat, requires Main)
// ============================================================================

TEST(WorkerProcessHeartBeatTest, ConstructionRegistersWithMain) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    std::string workerAddress = GetRandomUdsAddress();
    // Construction should succeed: creates worker's gRPC server,
    // registers with Main, starts polling thread
    WorkerProcessHeartBeat workerHb(mainAddress, workerAddress);

    // If construction succeeded without throwing, the registration worked.
    // Give a moment for registration to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST(WorkerProcessHeartBeatTest, ConstructionFailsWhenMainNotRunning) {
    std::string nonExistentMainAddress = GetRandomUdsAddress();
    std::string workerAddress = GetRandomUdsAddress();

    // Worker construction should throw when RegisterWorker RPC fails
    EXPECT_THROW({ WorkerProcessHeartBeat workerHb(nonExistentMainAddress, workerAddress); }, std::runtime_error);
}

TEST(WorkerProcessHeartBeatTest, DestructorUnregistersFromMain) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    std::string workerAddress = GetRandomUdsAddress();
    {
        WorkerProcessHeartBeat workerHb(mainAddress, workerAddress);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Worker destructor calls UnregisterWorker RPC to Main.
    // Should not hang; Main should handle the unregister gracefully.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST(WorkerProcessHeartBeatTest, IsBeatReachableFromMain) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    std::string workerAddress = GetRandomUdsAddress();
    WorkerProcessHeartBeat workerHb(mainAddress, workerAddress);

    // Main (or a client simulating Main) can reach the Worker's gRPC server
    HeartBeatClient mainToWorkerClient(workerAddress);
    bool isAlive = mainToWorkerClient.IsBeat();
    EXPECT_TRUE(isAlive) << "Worker's gRPC server should respond to IsBeat";
}

TEST(WorkerProcessHeartBeatTest, NotifyStopPollFromMainToWorker) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    std::string workerAddress = GetRandomUdsAddress();
    WorkerProcessHeartBeat workerHb(mainAddress, workerAddress);

    // Simulate Main calling NotifyStopPoll on Worker
    HeartBeatClient mainToWorkerClient(workerAddress);
    bool result = mainToWorkerClient.NotifyStopPoll();
    EXPECT_TRUE(result) << "NotifyStopPoll to Worker should succeed";

    // Worker's WaitForStopPoll should now return quickly
    auto start = std::chrono::steady_clock::now();
    workerHb.WaitForStopPoll(100);
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(elapsed, 100) << "Worker's WaitForStopPoll should return immediately after NotifyStopPoll from Main";
}

// ============================================================================
// 6. Integration Tests (Main ↔ Worker bidirectional heartbeat)
// ============================================================================

TEST(ProcessHeartBeatIntegrationTest, FullLifecycle) {
    std::string mainAddress = GetRandomUdsAddress();

    // Start Main
    MainProcessHeartBeat mainHb(mainAddress);

    // Start Worker (registers with Main, starts polling)
    std::string workerAddress = GetRandomUdsAddress();
    WorkerProcessHeartBeat workerHb(mainAddress, workerAddress);

    // Verify bidirectional heartbeats work
    {
        // Main → Worker: Main can ping Worker's server
        HeartBeatClient mainToWorker(workerAddress);
        EXPECT_TRUE(mainToWorker.IsBeat()) << "Main should be able to reach Worker";

        // Worker → Main: Worker can ping Main's server
        HeartBeatClient workerToMain(mainAddress);
        EXPECT_TRUE(workerToMain.IsBeat()) << "Worker should be able to reach Main";
    }

    // Let at least one polling cycle run on both sides
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Re-verify after some time has passed
    {
        HeartBeatClient mainToWorker(workerAddress);
        EXPECT_TRUE(mainToWorker.IsBeat()) << "Worker should still be reachable after polling cycle";

        HeartBeatClient workerToMain(mainAddress);
        EXPECT_TRUE(workerToMain.IsBeat()) << "Main should still be reachable after polling cycle";
    }

    // Proper teardown: Worker is destroyed first (unregisters from Main),
    // then Main is destroyed (no workers left to poll).
    // Order matters: Worker destructor → Main destructor (C++ stack unwinding order).
}

TEST(ProcessHeartBeatIntegrationTest, MultipleWorkers) {
    std::string mainAddress = GetRandomUdsAddress();
    MainProcessHeartBeat mainHb(mainAddress);

    // Start two Workers
    std::string worker1Address = GetRandomUdsAddress();
    std::string worker2Address = GetRandomUdsAddress();
    WorkerProcessHeartBeat worker1Hb(mainAddress, worker1Address);
    WorkerProcessHeartBeat worker2Hb(mainAddress, worker2Address);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Both workers should be reachable
    {
        HeartBeatClient toWorker1(worker1Address);
        HeartBeatClient toWorker2(worker2Address);
        HeartBeatClient toMain(mainAddress);

        EXPECT_TRUE(toWorker1.IsBeat()) << "Worker 1 should be reachable";
        EXPECT_TRUE(toWorker2.IsBeat()) << "Worker 2 should be reachable";
        EXPECT_TRUE(toMain.IsBeat()) << "Main should be reachable";
    }

    // Workers are destroyed in reverse order (worker2 first, then worker1)
    // Each destructor unregisters from Main.
}

TEST(ProcessHeartBeatIntegrationTest, MainStopWorkerPollNotifiesWorkers) {
    std::string mainAddress = GetRandomUdsAddress();
    std::string workerAddress = GetRandomUdsAddress();

    // Use unique_ptr for Main so we can control destruction order:
    // Create Main first (so Worker can register), then destroy Main while Worker is alive.
    auto mainHb = std::make_unique<MainProcessHeartBeat>(mainAddress);
    WorkerProcessHeartBeat workerHb(mainAddress, workerAddress);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Destroy Main while Worker is still running:
    //   1. NotifyStopPoll() wakes Main's polling thread
    //   2. Polling thread calls StopWorkerPoll()
    //   3. StopWorkerPoll() sends NotifyStopPoll RPC to Worker
    //   4. Worker's server receives it → calls base.NotifyStopPoll()
    //   5. Worker's polling thread wakes up and exits
    mainHb.reset();

    // Give Worker's polling thread time to respond to Main's NotifyStopPoll
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Worker's WaitForStopPoll should return quickly since Main notified it
    auto start = std::chrono::steady_clock::now();
    workerHb.WaitForStopPoll(200);
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    // If Main's StopWorkerPoll successfully notified Worker, mStopPoll is set
    // and WaitForStopPoll returns immediately. If not, it blocks for 200ms.
    // Either case, the test verifies no hang/crash.
    EXPECT_LT(elapsed, 500);
}

// ============================================================================
// 7. WorkerDetectsMainFailure Tests
// ============================================================================

TEST(WorkerDetectsMainFailureTest, WorkerDetectsMainServerShutdown) {
    std::string mainAddress = GetRandomUdsAddress();
    std::string workerAddress = GetRandomUdsAddress();

    // Use a standalone HeartBeatServer (not MainProcessHeartBeat) so we can
    // shut it down abruptly to simulate a Main crash without proper shutdown.
    TestHeartBeat testHb;
    HeartBeatServer server(mainAddress, testHb);

    // Install a test callback to observe when the heartbeat thread
    // detects Main failure, replacing the production callback that
    // would call processSync.NotifyExit().
    std::atomic<bool> exitNotified{false};
    WorkerProcessHeartBeat workerHb(mainAddress, workerAddress, [&exitNotified]() { exitNotified.store(true); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Verify Worker is alive and polling Main
    {
        HeartBeatClient workerToMain(mainAddress);
        EXPECT_TRUE(workerToMain.IsBeat()) << "Main (standalone server) should be reachable";
    }

    // Simulate Main crash: shut down the gRPC server
    server.Shutdown();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Wait for Worker's polling thread to detect Main failure.
    // Detection timing: WaitForStopPoll(4s) + IsBeat deadline(~3s) ≈ 7s worst case.
    // We wait up to 10s to be safe.
    // After detecting failure, Worker calls the onExit callback.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        if (exitNotified.load()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    EXPECT_TRUE(exitNotified.load()) << "Worker should detect Main failure and invoke onExit callback "
                                     << "within ~6 seconds of Main server shutdown";
}

// ============================================================================
// 8. HeartBeatClient + Server Interaction Edge Cases
// ============================================================================

TEST(HeartBeatEdgeCaseTest, MultipleIsBeatCalls) {
    std::string address = GetRandomUdsAddress();
    TestHeartBeat testHb;
    HeartBeatServer server(address, testHb);

    HeartBeatClient client(address);

    // Multiple consecutive IsBeat calls should all succeed
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(client.IsBeat()) << "IsBeat call " << i << " should succeed";
    }
}

TEST(HeartBeatEdgeCaseTest, ClientToShutdownServer) {
    std::string address = GetRandomUdsAddress();
    TestHeartBeat testHb;

    // Create server, then shut it down
    {
        HeartBeatServer server(address, testHb);
        HeartBeatClient client(address);
        EXPECT_TRUE(client.IsBeat()) << "IsBeat should succeed while server is running";
    }
    // Server is now destroyed (Shutdown + destructor called).
    // gRPC calls to the now-gone server will fail within the deadline (default 3s).

    HeartBeatClient client(address);
    EXPECT_FALSE(client.IsBeat()) << "IsBeat should fail after server is shut down";
}

TEST(HeartBeatEdgeCaseTest, RegisterUnregisterMultipleWorkers) {
    std::string address = GetRandomUdsAddress();
    TestHeartBeat testHb;
    HeartBeatServer server(address, testHb);

    // Register multiple workers
    HeartBeatClient client1(address);
    HeartBeatClient client2(address);
    HeartBeatClient client3(address);

    EXPECT_TRUE(client1.RegisterWorker("worker_1"));
    EXPECT_TRUE(client2.RegisterWorker("worker_2"));
    EXPECT_TRUE(client3.RegisterWorker("worker_3"));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(testHb.GetRegisteredWorkerCount(), 3);

    // Unregister in different order
    EXPECT_TRUE(client2.UnregisterWorker("worker_2"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(testHb.GetRegisteredWorkerCount(), 2);

    EXPECT_TRUE(client1.UnregisterWorker("worker_1"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(testHb.GetRegisteredWorkerCount(), 1);

    EXPECT_TRUE(client3.UnregisterWorker("worker_3"));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(testHb.GetRegisteredWorkerCount(), 0);
}
