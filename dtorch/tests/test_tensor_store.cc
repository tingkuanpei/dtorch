/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <torch/torch.h>

#include "dtorch/api/cpp/device.h"
#include "dtorch/common/argument_parser.h"
#include "dtorch/common/process/sub_process.h"
#include "dtorch/core/communication/tensor_store/tensor_store.h"
#include "dtorch/external/boost/boost_interprocess.h"
#include "dtorch/external/device/device_stream.h"
#include "dtorch/external/torch/torch_util.h"
#include "test.h"

using namespace dtorch::core::communication;
using namespace dtorch::api::cpp;
using namespace dtorch::core;
using namespace dtorch::external::torch;
using namespace dtorch::external::boost;
using namespace dtorch::external::device;

void SetTensorMainFunc(TensorStoreType tensorStoreType, const Device& device, const std::string& storeKey) {
    const std::string kTensorKey0 = "TensorKey0";
    const std::string kTensorKey1 = "TensorKey1";
    DeviceStream stream = DeviceStream::GetCurrentStream(device);

    TensorStoreCreateInfo createInfo(TensorStoreConfig(tensorStoreType), storeKey, 2);
    auto tensorStore = TensorStore::Create(createInfo);
    tensorStore->Barrier();
    {
        tensorStore->SrcSet(kTensorKey0, torch::zeros({2, 2}).to(TorchUtil::ToDevice(device)), stream, 1);
        tensorStore->SrcSet(kTensorKey1, torch::ones({2, 2}).to(TorchUtil::ToDevice(device)), stream, 1);
        tensorStore->SrcWaitUntilGetFinished(kTensorKey0, stream);
        tensorStore->SrcWaitUntilGetFinished(kTensorKey1, stream);
    }

    tensorStore->Reset();

    {
        tensorStore->SrcSet(kTensorKey0, torch::zeros({2, 2}).to(TorchUtil::ToDevice(device)), stream, 1);
        tensorStore->SrcSet(kTensorKey1, torch::ones({2, 2}).to(TorchUtil::ToDevice(device)), stream, 1);
        tensorStore->SrcWaitUntilGetFinished(kTensorKey0, stream);
        tensorStore->SrcWaitUntilGetFinished(kTensorKey1, stream);
    }
    tensorStore->Barrier();
}

void GetTensorMainFunc(TensorStoreType tensorStoreType, const Device& device, const std::string& storeKey) {
    const std::string kTensorKey0 = "TensorKey0";
    const std::string kTensorKey1 = "TensorKey1";
    DeviceStream stream = DeviceStream::GetCurrentStream(device);

    TensorStoreCreateInfo createInfo(TensorStoreConfig(tensorStoreType), storeKey, 2);
    auto tensorStore = TensorStore::Create(createInfo);
    tensorStore->Barrier();
    {
        auto tensor0 = tensorStore->DestGet(kTensorKey0, stream);
        auto tensor1 = tensorStore->DestGet(kTensorKey1, stream);
        EXPECT_TRUE(torch::allclose(tensor0, torch::zeros({2, 2}).to(TorchUtil::ToDevice(device))));
        EXPECT_TRUE(torch::allclose(tensor1, torch::ones({2, 2}).to(TorchUtil::ToDevice(device))));
        tensorStore->DestFinishGet(kTensorKey0, stream);
        tensorStore->DestFinishGet(kTensorKey1, stream);
    }

    tensorStore->Reset();

    {
        auto tensor0 = tensorStore->DestGet(kTensorKey0, stream);
        auto tensor1 = tensorStore->DestGet(kTensorKey1, stream);
        EXPECT_TRUE(torch::allclose(tensor0, torch::zeros({2, 2}).to(TorchUtil::ToDevice(device))));
        EXPECT_TRUE(torch::allclose(tensor1, torch::ones({2, 2}).to(TorchUtil::ToDevice(device))));
        tensorStore->DestFinishGet(kTensorKey0, stream);
        tensorStore->DestFinishGet(kTensorKey1, stream);
    }
    tensorStore->Barrier();
}

void SetTensorCrossDeviceMainFunc(TensorStoreType tensorStoreType, const Device& srcDevice,
                                  DeviceKind destGetDeviceKind, const std::string& storeKey) {
    const std::string kTensorKey0 = "TensorKey0";
    const std::string kTensorKey1 = "TensorKey1";
    DeviceStream stream = DeviceStream::GetCurrentStream(srcDevice);

    TensorStoreCreateInfo createInfo(TensorStoreConfig(tensorStoreType), storeKey, 2);
    auto tensorStore = TensorStore::Create(createInfo);
    tensorStore->Barrier();
    {
        tensorStore->SrcSet(kTensorKey0, torch::zeros({2, 2}).to(TorchUtil::ToDevice(srcDevice)), stream, 1,
                            destGetDeviceKind);
        tensorStore->SrcSet(kTensorKey1, torch::ones({2, 2}).to(TorchUtil::ToDevice(srcDevice)), stream, 1,
                            destGetDeviceKind);
        tensorStore->SrcWaitUntilGetFinished(kTensorKey0, stream);
        tensorStore->SrcWaitUntilGetFinished(kTensorKey1, stream);
    }
    tensorStore->Barrier();
}

void GetTensorCrossDeviceMainFunc(TensorStoreType tensorStoreType, const Device& destDevice,
                                  const std::string& storeKey) {
    const std::string kTensorKey0 = "TensorKey0";
    const std::string kTensorKey1 = "TensorKey1";
    DeviceStream stream = DeviceStream::GetCurrentStream(destDevice);

    TensorStoreCreateInfo createInfo(TensorStoreConfig(tensorStoreType), storeKey, 2);
    auto tensorStore = TensorStore::Create(createInfo);
    tensorStore->Barrier();
    {
        auto tensor0 = tensorStore->DestGet(kTensorKey0, stream);
        auto tensor1 = tensorStore->DestGet(kTensorKey1, stream);
        EXPECT_TRUE(torch::allclose(tensor0, torch::zeros({2, 2}).to(TorchUtil::ToDevice(destDevice))));
        EXPECT_TRUE(torch::allclose(tensor1, torch::ones({2, 2}).to(TorchUtil::ToDevice(destDevice))));
        EXPECT_EQ(TorchUtil::GetDevice(tensor0).deviceKind, destDevice.deviceKind);
        EXPECT_EQ(TorchUtil::GetDevice(tensor1).deviceKind, destDevice.deviceKind);
        tensorStore->DestFinishGet(kTensorKey0, stream);
        tensorStore->DestFinishGet(kTensorKey1, stream);
    }
    tensorStore->Barrier();
}

void TestMultiThreadImp(TensorStoreType tensorStoreType, const Device& device) {
    const std::string storeKey = ManagedSharedMemory::GetShmFileNameWithPrefix("TensorStoreMultiThreadTest");

    std::thread setTensorThread(SetTensorMainFunc, tensorStoreType, device, storeKey);
    std::thread getTensorThread(GetTensorMainFunc, tensorStoreType, device, storeKey);
    setTensorThread.join();
    getTensorThread.join();
}

void TestMultiThreadCrossDeviceImp(TensorStoreType tensorStoreType, const Device& srcDevice, const Device& destDevice) {
    const std::string storeKey = ManagedSharedMemory::GetShmFileNameWithPrefix("TensorStoreMultiThreadCrossDeviceTest");

    std::thread setTensorThread(SetTensorCrossDeviceMainFunc, tensorStoreType, srcDevice, destDevice.deviceKind,
                                storeKey);
    std::thread getTensorThread(GetTensorCrossDeviceMainFunc, tensorStoreType, destDevice, storeKey);
    setTensorThread.join();
    getTensorThread.join();
}

TEST(TensorStoreTest, MultiThreadTest) {
    TestMultiThreadImp(TensorStoreType::kMemory, Device(DeviceKind::kCpu, 0));
    TestMultiThreadImp(TensorStoreType::kFile, Device(DeviceKind::kCpu, 0));

    if (Device::IsAvailable(DeviceKind::kGpu)) {
        TestMultiThreadImp(TensorStoreType::kMemory, Device(DeviceKind::kGpu, 0));
        TestMultiThreadImp(TensorStoreType::kFile, Device(DeviceKind::kGpu, 0));
    }
}

TEST(TensorStoreTest, MultiThreadCrossDeviceTest) {
    if (!Device::IsAvailable(DeviceKind::kGpu)) {
        return;
    }

    TestMultiThreadCrossDeviceImp(TensorStoreType::kMemory, Device(DeviceKind::kCpu, 0), Device(DeviceKind::kGpu, 0));
    TestMultiThreadCrossDeviceImp(TensorStoreType::kFile, Device(DeviceKind::kCpu, 0), Device(DeviceKind::kGpu, 0));

    TestMultiThreadCrossDeviceImp(TensorStoreType::kMemory, Device(DeviceKind::kGpu, 0), Device(DeviceKind::kCpu, 0));
    TestMultiThreadCrossDeviceImp(TensorStoreType::kFile, Device(DeviceKind::kGpu, 0), Device(DeviceKind::kCpu, 0));
}

TEST(TensorStoreTest, MultiProcessTest) {
    const auto& parser = dtorch::ArgumentParser::GetSingleton();
    if (parser.HasOption("child")) {
        const std::string storeKey = parser.OptionValue("store_key");
        // GetTensorMainFunc(TensorStoreType::kFile, Device(DeviceKind::kCpu, 0), storeKey);
        GetTensorMainFunc(TensorStoreType::kFile, Device(DeviceKind::kGpu, 0), storeKey);
    } else {
        const std::string storeKey = ManagedSharedMemory::GetShmFileNameWithPrefix("TensorStoreMultiThreadTest");

        // Launch child process
        const std::string exeCmd = parser.ProgramName() + " --gtest_filter=TensorStoreTest.MultiProcessTest --child" +
                                   " --store_key=" + storeKey;
        dtorch::SubProcess subProcess(exeCmd);
        dtorch::IgnoreUnused(subProcess);

        // SetTensorMainFunc(TensorStoreType::kFile, Device(DeviceKind::kCpu, 0), storeKey);
        SetTensorMainFunc(TensorStoreType::kFile, Device(DeviceKind::kGpu, 0), storeKey);
    }
}

TEST(TensorStoreTest, MultiProcessCrossDeviceTest) {
    const auto& parser = dtorch::ArgumentParser::GetSingleton();
    if (parser.HasOption("child")) {
        if (!Device::IsAvailable(DeviceKind::kGpu)) {
            return;
        }
        const std::string cpu2gpuStoreKey = parser.OptionValue("store_key_cpu2gpu");
        const std::string gpu2cpuStoreKey = parser.OptionValue("store_key_gpu2cpu");
        GetTensorCrossDeviceMainFunc(TensorStoreType::kFile, Device(DeviceKind::kGpu, 0), cpu2gpuStoreKey);
        GetTensorCrossDeviceMainFunc(TensorStoreType::kFile, Device(DeviceKind::kCpu, 0), gpu2cpuStoreKey);
    } else {
        if (!Device::IsAvailable(DeviceKind::kGpu)) {
            return;
        }

        const std::string cpu2gpuStoreKey =
            ManagedSharedMemory::GetShmFileNameWithPrefix("TensorStoreMultiProcessCrossDeviceCpu2GpuTest");
        const std::string gpu2cpuStoreKey =
            ManagedSharedMemory::GetShmFileNameWithPrefix("TensorStoreMultiProcessCrossDeviceGpu2CpuTest");

        // Launch child process
        const std::string exeCmd =
            parser.ProgramName() + " --gtest_filter=TensorStoreTest.MultiProcessCrossDeviceTest --child" +
            " --store_key_cpu2gpu=" + cpu2gpuStoreKey + " --store_key_gpu2cpu=" + gpu2cpuStoreKey;
        dtorch::SubProcess subProcess(exeCmd);
        dtorch::IgnoreUnused(subProcess);

        SetTensorCrossDeviceMainFunc(TensorStoreType::kFile, Device(DeviceKind::kCpu, 0), DeviceKind::kGpu,
                                     cpu2gpuStoreKey);
        SetTensorCrossDeviceMainFunc(TensorStoreType::kFile, Device(DeviceKind::kGpu, 0), DeviceKind::kCpu,
                                     gpu2cpuStoreKey);
    }
}
