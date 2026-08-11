/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <cmath>
#include <stdexcept>

#include "dtorch/api/cpp/device.h"
#include "dtorch/common/config.h"
#include "dtorch/external/device/device_peak_performance.h"
#include "test.h"

using namespace dtorch::api::cpp;
using namespace dtorch::external::device;

// ============================================================
// CPU tests (always available, no CUDA needed)
// ============================================================

TEST(DevicePeakPerformanceTest, CpuReturnsNullopt) {
    DevicePeakPerformance perf(Device::GetDefaultCpuDevice());
    EXPECT_EQ(perf.GetDevice(), Device::GetDefaultCpuDevice());

    // CPU is not implemented yet — all getters return nullopt.
    EXPECT_FALSE(perf.GetMemoryBandwidth().has_value());
    EXPECT_FALSE(perf.GetFp32Flops().has_value());
    EXPECT_FALSE(perf.GetFp16Flops().has_value());
    EXPECT_FALSE(perf.GetBf16Flops().has_value());
    EXPECT_FALSE(perf.GetFp8Flops().has_value());
}

TEST(DevicePeakPerformanceTest, FactoryThrowsOnUnknownKind) {
    EXPECT_THROW({ DevicePeakPerformance perf(Device(DeviceKind::kCount, 0)); }, std::invalid_argument);
}

// ============================================================
// CUDA tests (only compiled and run when CUDA is available)
// ============================================================

#if DTORCH_WITH_CUDA

TEST(DevicePeakPerformanceTest, CudaReturnsPeaks) {
    DevicePeakPerformance perf(Device(DeviceKind::kGpu, 0));
    EXPECT_EQ(perf.GetDevice().deviceKind, DeviceKind::kGpu);
    EXPECT_EQ(perf.GetDevice().deviceId, 0);

    // Memory bandwidth: must be present, finite, positive, in [1 GB/s, 10 TB/s].
    auto bw = perf.GetMemoryBandwidth();
    ASSERT_TRUE(bw.has_value());
    EXPECT_TRUE(std::isfinite(*bw));
    EXPECT_GT(*bw, 1e9);
    EXPECT_LT(*bw, 1e13);

    // FP32: must be present, positive, in [10 GFLOPS, 10 PFLOPS].
    auto fp32 = perf.GetFp32Flops();
    ASSERT_TRUE(fp32.has_value());
    EXPECT_TRUE(std::isfinite(*fp32));
    EXPECT_GT(*fp32, 1e10);
    EXPECT_LT(*fp32, 1e16);

    // FP16 / BF16 / FP8: at minimum FP32 must be present; the others are
    // architecture-dependent, but for any GPU with SM ≥ 6.0 FP16 should exist.
    auto fp16 = perf.GetFp16Flops();
    ASSERT_TRUE(fp16.has_value());
    EXPECT_TRUE(std::isfinite(*fp16));
    EXPECT_GE(*fp16, *fp32);  // FP16 throughput ≥ FP32 on all GPUs that support it
}

TEST(DevicePeakPerformanceTest, CudaAllDevicesSupported) {
    int deviceCount = static_cast<int>(Device::DeviceCount(DeviceKind::kGpu));
    ASSERT_GT(deviceCount, 0);

    for (int i = 0; i < deviceCount; ++i) {
        DevicePeakPerformance perf(Device(DeviceKind::kGpu, i));
        EXPECT_EQ(perf.GetDevice().deviceId, i);

        // These three must always be present on any real GPU.
        EXPECT_TRUE(perf.GetMemoryBandwidth().has_value());
        EXPECT_TRUE(perf.GetFp32Flops().has_value());
        EXPECT_TRUE(perf.GetFp16Flops().has_value());  // SM ≥ 5.3; all real GPUs today

        // BF16 / FP8 presence depends on architecture — just check they don't crash.
        auto bf16 = perf.GetBf16Flops();
        auto fp8 = perf.GetFp8Flops();
        if (bf16.has_value()) {
            EXPECT_TRUE(std::isfinite(*bf16));
            EXPECT_GE(*bf16, *perf.GetFp32Flops());
        }
        if (fp8.has_value()) {
            EXPECT_TRUE(std::isfinite(*fp8));
            EXPECT_GE(*fp8, *perf.GetFp16Flops());
        }
    }
}

#endif  // DTORCH_WITH_CUDA
