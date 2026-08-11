/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <chrono>
#include <iostream>

#include <torch/torch.h>
#include <torch/types.h>

#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/dtorch.h"
#include "dtorch/external/torch/torch_util.h"
#include "test.h"

using namespace dtorch::api::cpp;
using namespace dtorch::core;
using namespace dtorch::external::torch;

TEST(SystemOverheadTest, BenchmarkAdd) {
    constexpr int kWarmup = 1000;
    constexpr int kMeasure = 20000;
    Device device(DeviceKind::kCpu);
    auto torchDevice = TorchUtil::ToDevice(device);

    // ---- Benchmark line 24: torch::Tensor torchTensorC = torchTensorA.add(torchTensorB) ----
    {
        torch::Tensor a = torch::rand({1, 1024}).to(torchDevice);
        torch::Tensor b = torch::rand({1, 1024}).to(torchDevice);

        // warmup
        for (int i = 0; i < kWarmup; ++i) {
            torch::Tensor c = a.add(b);
        }

        // measure
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kMeasure; ++i) {
            torch::Tensor c = a.add(b);  // line 24 equivalent
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double totalUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double avgUs = totalUs / kMeasure;
        std::cout << "[Benchmark] torch::Tensor::add (line 24): avg = " << avgUs << " us over " << kMeasure
                  << " iterations (warmup=" << kWarmup << ")" << std::endl;
    }

    // ---- Benchmark line 32: ug::Tensor tensorC = ug::functional::_Add(tensorA, tensorB) ----
    {
        torch::Tensor torchA = torch::rand({1, 1024}).to(torchDevice);
        torch::Tensor torchB = torch::rand({1, 1024}).to(torchDevice);

        GraphOption option;
        option.perDevicePerProcess = false;
        ug::Graph graph(option);
        ug::Tensor tensorA(graph, torchA);
        ug::Tensor tensorB(graph, torchB);

        // warmup
        for (int i = 0; i < kWarmup; ++i) {
            ug::Tensor tensorC = ug::functional::_Add(tensorA, tensorB);
        }

        // measure
        graph.Sync();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kMeasure; ++i) {
            ug::Tensor tensorC = ug::functional::_Add(tensorA, tensorB);  // line 32 equivalent
        }
        graph.Sync();
        auto t1 = std::chrono::high_resolution_clock::now();

        double totalUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double avgUs = totalUs / kMeasure;
        std::cout << "[Benchmark] ug::functional::_Add (line 32): avg = " << avgUs << " us over " << kMeasure
                  << " iterations (warmup=" << kWarmup << ")" << std::endl;
    }
}
