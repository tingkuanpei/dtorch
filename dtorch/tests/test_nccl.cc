/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/torch.h>

#include "dtorch/api/cpp/device.h"
#include "dtorch/common/filesystem.h"
#include "dtorch/core/communication/thread_group/thread_group.h"
#include "dtorch/external/cuda/cuda_device.h"
#include "test.h"

using namespace dtorch;
using namespace dtorch::external::cuda;
using namespace dtorch::core;
using namespace dtorch::api::cpp;

using FuncType = std::function<void(const std::string& initPath, int64_t, int64_t)>;

void TestAllreduce(const std::string& initPath, int64_t rank, int64_t size) {
    CudaDevice::SetDevice(rank);
    c10::cuda::CUDAStream stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);

    at::Tensor input = at::ones({1000, 1000}, at::kCUDA);
    torch::Tensor expectResult = at::ones({1000, 1000}, at::kCUDA) * size;

    std::vector<int64_t> allGlobalDeviceId;
    for (int64_t i = 0; i < size; i++) {
        allGlobalDeviceId.push_back(i);
    }

    communication::ThreadGroup threadGroup(initPath, DeviceKind::kGpu, allGlobalDeviceId, rank, size, false);
    threadGroup.Barrier();
    at::Tensor output = threadGroup.AllReduce(input);
    threadGroup.Barrier();

    EXPECT_TRUE(torch::allclose(output, expectResult));
}

void MultiThreadRun(FuncType testFunc) {
    int64_t numGpus = Device::DeviceCount(DeviceKind::kGpu);
    if (numGpus <= 1) {
        return;
    }

    std::string initPath = "TestNcclMultiThread";
    std::vector<std::thread> threads;
    threads.reserve(numGpus);

    for (int64_t i = 0; i < numGpus; i++) {
        threads.emplace_back(std::thread(testFunc, initPath, i, numGpus));
    }

    for (int64_t i = 0; i < numGpus; i++) {
        threads[i].join();
    }
}

TEST(NcclTest, SimpleTest) {
    if (!Device::IsAvailable(DeviceKind::kGpu)) {
        return;
    }

    MultiThreadRun(TestAllreduce);
}
