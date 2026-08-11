/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "remote_runner_in_process.h"

#include <memory>

#include "dtorch/api/cpp/cluster.h"
#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/argument_parser.h"
#include "dtorch/common/filesystem.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/communication/global_instance_id.h"
#include "dtorch/core/distributed/cluster_info.h"
#include "dtorch/core/distributed/process_heart_beat.h"
#include "dtorch/core/runner/remote/remote_runner.h"
#include "dtorch/external/boost/boost_interprocess.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/python/python_gil.h"
#include "dtorch/external/rpc/rpc_common.h"
#include "dtorch/external/zmq/zmq.h"

namespace dtorch {
namespace core {

RemoteRunnerInProcess::RemoteRunnerInProcess(const GraphOption& graphOption,
                                             const RunnerSupportedDevices& supportedDevices,
                                             const std::string& publisherAddress,
                                             const std::string& mainPushPullAddress)
    : processSync(), subprocess() {
    std::pair<SubProcess, SubProcessSync> result = RemoteRunnerInProcessLauncher::StartRemoteRunnerInBackgroundProcess(
        publisherAddress, mainPushPullAddress, graphOption, supportedDevices);
    subprocess = std::move(result.first);
    processSync = std::move(result.second);
}

void RemoteRunnerInProcess::WaitSubProcessStarted() {
    DDebugAssert(processSync.IsCreated());
    processSync.WaitProcessStarted();
}

void RemoteRunnerInProcess::NotifySubProcessExit() {
    if (processSync.IsCreated()) {
        processSync.NotifyExit();
    }
}

RemoteRunnerInProcess::~RemoteRunnerInProcess() { NotifySubProcessExit(); }

std::pair<SubProcess, SubProcessSync> RemoteRunnerInProcessLauncher::StartRemoteRunnerInBackgroundProcess(
    const std::string& publisherAddress, const std::string& mainPushPullAddress, const GraphOption& graphOption,
    const RunnerSupportedDevices& supportedDevices) {
    // 1. Serialize argument
    std::stringstream ss(std::ios::out | std::ios::binary);
    external::boost::BinaryOArchive boa(ss);
    DAlwaysAssert(api::cpp::distributed::Cluster::GetSingleton().IsCreate());
    std::string globalOptionData = core::GlobalOption::GetSingleton().SerializeToString();
    boa << globalOptionData;
    const std::string mainProcessHeartBeatAddress =
        api::cpp::distributed::Cluster::GetSingleton().GetMainProcessHeartBeatAddress();
    boa << mainProcessHeartBeatAddress;
    boa << publisherAddress;
    boa << mainPushPullAddress;
    boa << graphOption;
    boa << supportedDevices;
    std::string instanceId = communication::GlobalCommInstanceId::GetSingleton().GetInstanceId();
    boa << instanceId;
    // Ship the cluster-wide ClusterInfo so the subprocess can validate distributed specs
    // against the true total GPU count (it has no Cluster context of its own).
    std::string clusterInfoData = distributed::ClusterInfo::GetSingleton().SerializeToString();
    boa << clusterInfoData;

    // 2. Write argument to shared memory
    const std::string shmFileName = external::boost::ManagedSharedMemory::GetShmFileNameWithPrefix(GetRandomFileName());
    SubProcessSync processSync(shmFileName, true);
    processSync.SetLaunchProcessArgument(ss.str());

    // 3. Launch child process
    //
    // CUDA does NOT support fork() after CUDA context creation — the child would inherit
    // the parent's CUDA state, causing undefined behavior (typically crashes or hangs).
    // Since the parent process (Python) has already initialized LibTorch/CUDA, we MUST
    // use exec() to launch a fresh executable, which re-initializes CUDA from scratch.
    std::stringstream css;
    css << "dtorch_launcher --remote-runner";
    css << " --shm_file_name=" << shmFileName;

    SubProcess subProcess(css.str());

    return std::make_pair(std::move(subProcess), std::move(processSync));
}

int RemoteRunnerInProcessLauncher::BackgroundProcessExecMain(const std::vector<std::string>& arguments) {
    // 1. Prepare environment
    // Release Python GIL to avoid deadlock.
    auto scopedRelease = external::python::GetPythonGilScopedRelease();

    // 2. Open shared memory
    auto& parser = ArgumentParser::GetSingleton();
    parser.Init(arguments);
    DAlwaysAssertMsg(parser.HasOption("shm_file_name"), "Can't run this command from terminal directly");
    const std::string shmFileName = parser.OptionValue("shm_file_name");
    SubProcessSync processSync(shmFileName, false);
    auto argumentStr = processSync.GetLaunchProcessArgument();

    // 3. Get argument from shared memory
    std::stringstream ss(argumentStr, std::ios::in | std::ios::binary);
    external::boost::BinaryIArchive bia(ss);
    std::string globalOptionData;
    bia >> globalOptionData;
    core::GlobalOption::GlobalOptionInitFromString(globalOptionData);
    std::string mainProcessHeartBeatAddress;
    bia >> mainProcessHeartBeatAddress;
    std::string publisherAddress;
    std::string mainPushPullAddress;
    GraphOption graphOption;
    RunnerSupportedDevices supportedDevices;
    std::string instanceId;
    bia >> publisherAddress;
    bia >> mainPushPullAddress;
    bia >> graphOption;
    bia >> supportedDevices;
    bia >> instanceId;
    std::string clusterInfoData;
    bia >> clusterInfoData;

    { communication::GlobalCommInstanceId::GetSingleton().SetInstanceId(instanceId); }

    // Populate the ClusterInfo singleton before constructing the RemoteRunner: the runner will
    // re-Infer operators (CheckInputDistributedSpec), which needs the cluster-wide total GPU count.
    distributed::ClusterInfo::InitFromString(clusterInfoData);

    // 4. Start RemoteRunner
    RemoteRunner runner(mainPushPullAddress, publisherAddress, graphOption, supportedDevices,
                        communication::TensorStoreConfig(communication::TensorStoreType::kFile));

    // 5. Notify parent process
    processSync.NotifyProcessStarted();

    // 6. Start worker process heart beat.
    // Pass an exit callback so the heartbeat thread can wake us from
    // WaitForExit() when Main dies, rather than blocking forever.
    std::string workerHeartBeatAddress = external::rpc::GetRandomUdsAddress();
    std::unique_ptr<WorkerProcessHeartBeat> workerProcessHeartBeat = nullptr;
    try {
        workerProcessHeartBeat = std::make_unique<WorkerProcessHeartBeat>(
            mainProcessHeartBeatAddress, workerHeartBeatAddress, [&processSync]() { processSync.NotifyExit(); });
    } catch (const std::runtime_error& e) {
        DLogError() << "BackgroundProcessExecMain start WorkerProcessHeartBeat failed: " << e.what();
        return 1;
    }

    // 7. Block until the parent signals exit via shared memory.
    processSync.WaitForExit();

    return 0;
}

}  // namespace core
}  // namespace dtorch
