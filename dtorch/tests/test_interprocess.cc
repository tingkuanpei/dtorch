/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <cassert>
#include <cstddef>
#include <utility>

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "dtorch/common/argument_parser.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/process/sub_process.h"
#include "dtorch/external/boost/boost_interprocess.h"
#include "dtorch/external/cuda/cuda_device.h"
#include "dtorch/external/cuda/cuda_error.h"
#include "dtorch/external/torch/torch_util.h"
#include "test.h"

using namespace dtorch::api::cpp;
using namespace dtorch::core;
using namespace dtorch::external::boost;
using namespace dtorch::external::torch;
using namespace dtorch::external::cuda;

TEST(InterprocessTest, SimpleTest) {
    using PairType = std::pair<double, int>;

    const auto &parser = dtorch::ArgumentParser::GetSingleton();
    if (parser.HasOption("child")) {
        const std::string shmFileName = parser.OptionValue("shm_file_name");
        ManagedSharedMemory memory(::boost::interprocess::open_only, shmFileName);

        auto array = memory.Find<PairType>("Array", 10);
        EXPECT_TRUE(array[1].first == 2.0);
        EXPECT_TRUE(array[1].second == 4);
        auto instance = memory.Find<PairType>("Instance");
        EXPECT_TRUE(instance->first == 1.0);
        EXPECT_TRUE(instance->second == 3);
        auto str = memory.FindStr("String");
        EXPECT_TRUE(str == "This is string content");

        ShmStringAllocator stringAlloc(memory.mMemory->get_segment_manager());
        ShmStringSet *stringSet = memory.FindStringSet("StringSet");
        EXPECT_TRUE(stringSet->find(ShmString("apple", stringAlloc)) != stringSet->end());
        EXPECT_TRUE(stringSet->find(ShmString("banana", stringAlloc)) != stringSet->end());
        EXPECT_TRUE(stringSet->find(ShmString("cherry", stringAlloc)) != stringSet->end());
        EXPECT_TRUE(stringSet->find(ShmString("dog", stringAlloc)) == stringSet->end());
        EXPECT_TRUE(stringSet->size() == 3);

        memory.Destroy<PairType>("Array");
        memory.Destroy<PairType>("Instance");

        *(memory.Find<bool>("Flag")) = true;
        memory.Find<InterprocessCondition>("Cond")->notify_all();
    } else {
        const std::string shmFileName = ManagedSharedMemory::GetShmFileNameWithPrefix("SimpleTest");
        ShmAutoRemove remover(shmFileName);
        ManagedSharedMemory memory(::boost::interprocess::create_only, shmFileName, 9012);

        memory.Construct<PairType>("Instance", 1.0, 3);
        memory.ConstructArray<PairType>("Array", 10, 2.0, 4);
        memory.ConstructString("String", "This is string content");
        auto *flag = memory.Construct<bool>("Flag", false);
        auto *mutex = memory.Construct<InterprocessMutex>("Mutex");
        auto *cv = memory.Construct<InterprocessCondition>("Cond");

        ShmStringAllocator stringAlloc(memory.mMemory->get_segment_manager());
        ShmStringSet *stringSet = memory.ConstructStringSet("StringSet", stringAlloc);
        stringSet->insert(ShmString("apple", stringAlloc));
        stringSet->insert(ShmString("banana", stringAlloc));
        stringSet->insert(ShmString("cherry", stringAlloc));

        // Launch child process
        const std::string exeCmd = parser.ProgramName() + " --gtest_filter=InterprocessTest.SimpleTest --child" +
                                   " --shm_file_name=" + shmFileName;
        auto subProcess = std::make_unique<dtorch::SubProcess>(exeCmd);

        ScopedLock<InterprocessMutex> lock(*mutex);
        cv->wait(lock, [=]() { return *flag; });
        EXPECT_TRUE(*flag == true);
        EXPECT_TRUE(memory.Count<PairType>("instance") == 0 && memory.Count<PairType>("array") == 0);

        subProcess.reset();
    }
}

// CUDA IPC not support in windows and WSL
// https://docs.pytorch.org/docs/stable/notes/windows.html#cuda-ipc-operations
TEST(InterprocessTest, CudaIpcTest) {
    if (!Device::IsAvailable(DeviceKind::kGpu) || !CudaDevice::IsSupportIpc(0)) {
        return;
    }

    const auto &parser = dtorch::ArgumentParser::GetSingleton();
    if (parser.HasOption("child")) {
        // Child Process
        const std::string shmFileName = parser.OptionValue("shm_file_name");
        ManagedSharedMemory memory(::boost::interprocess::open_only, shmFileName);
        std::string ipcHandleStr = memory.FindStr("CudaIpcMem instance");

        torch::Tensor fromHandleTensor = TorchUtil::FromIpcMemHandle(ipcHandleStr);
        torch::Tensor expectTensor = torch::eye(3).to(TorchUtil::ToDevice(DeviceKind::kGpu)).chunk(3, 1)[1];
        EXPECT_TRUE(torch::allclose(fromHandleTensor, expectTensor));
    } else {
        // Parent Process
        // Create a tensor with stride
        torch::Tensor torchTensorA = torch::eye(3).to(TorchUtil::ToDevice(DeviceKind::kGpu)).chunk(3, 1)[1];
        at::cuda::getCurrentCUDAStream().synchronize();
        std::string ipcHandleStr = TorchUtil::ToIpcMemHandle(torchTensorA);

        const std::string shmFileName = ManagedSharedMemory::GetShmFileNameWithPrefix("CudaIpcTest");
        ShmAutoRemove remover(shmFileName);
        ManagedSharedMemory memory(::boost::interprocess::create_only, shmFileName, 4096);
        memory.ConstructString("CudaIpcMem instance", ipcHandleStr);

        // Launch child process
        const std::string exeCmd = parser.ProgramName() + " --gtest_filter=InterprocessTest.CudaIpcTest --child" +
                                   " --shm_file_name=" + shmFileName;
        dtorch::SubProcess subProcess(exeCmd);
        dtorch::IgnoreUnused(subProcess);
    }
}
