/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/common/process/sub_process.h"
#include "dtorch/common/process/sub_process_sync.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/runner/runner_supported_devices.h"

namespace dtorch {
namespace core {

class RemoteRunnerInProcessLauncher {
public:
    // Called in the parent (Controller) process. Launches a child process via fork/exec
    // that runs BackgroundProcessExecMain. The child process creates a RemoteRunner and
    // communicates with the parent via ZMQ. Returns the child process handle and shared
    // memory sync object.
    static std::pair<SubProcess, SubProcessSync> StartRemoteRunnerInBackgroundProcess(
        const std::string& publisherAddress, const std::string& mainPushPullAddress, const GraphOption& graphOption,
        const RunnerSupportedDevices& supportedDevices);

    // Child process entry point. Reads launch arguments from shared memory, creates a
    // RemoteRunner, and notifies the parent via SubProcessSync. The child process main
    // loop runs in RemoteRunner::AsyncMain.
    static int BackgroundProcessExecMain(const std::vector<std::string>& arguments);
};

struct RemoteRunnerInProcess {
    SubProcessSync processSync;
    SubProcess subprocess;

public:
    RemoteRunnerInProcess(const GraphOption& graphOption, const RunnerSupportedDevices& supportedDevices,
                          const std::string& publisherAddress, const std::string& mainPushPullAddress);

    void WaitSubProcessStarted();

    // Notify the child process to exit (via shared memory condition variable).
    // Call this BEFORE the destructor to notify all subprocesses concurrently,
    // rather than letting each destructor notify sequentially.
    void NotifySubProcessExit();

    // Notify the child process to exit (via shared memory condition variable),
    // then ~SubProcess() blocks until the child has exited.
    ~RemoteRunnerInProcess();

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(RemoteRunnerInProcess);
};

}  // namespace core
}  // namespace dtorch
