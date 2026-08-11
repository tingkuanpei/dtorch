/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "dtorch/common/debug.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace external {
namespace device {

// Hardware-side half of the OperatorCost roofline analysis:
// theoretical peak throughput ceilings per device.
//
// Construct with a core::Device to query the device's peak capabilities.
// For CPU, all getters return std::nullopt (not yet implemented).
// For GPU, the constructor snapshots cudaDeviceProp and computes all peaks
// immediately, so subsequent getter calls are trivial.

class DevicePeakPerformance {
public:
    explicit DevicePeakPerformance(const core::Device& device);

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(DevicePeakPerformance);

    DTORCH_FORCEINLINE const core::Device& GetDevice() const noexcept { return mDevice; }

    // Theoretical peak memory bandwidth (bytes/second).
    DTORCH_FORCEINLINE std::optional<double> GetMemoryBandwidth() const { return mMemoryBandwidth; }

    // Theoretical peak FP32 compute throughput (FLOPs/second, FMA = 2 FLOPs).
    DTORCH_FORCEINLINE std::optional<double> GetFp32Flops() const { return mFp32Flops; }

    DTORCH_FORCEINLINE std::optional<double> GetFp16Flops() const { return mFp16Flops; }

    DTORCH_FORCEINLINE std::optional<double> GetBf16Flops() const { return mBf16Flops; }

    DTORCH_FORCEINLINE std::optional<double> GetFp8Flops() const { return mFp8Flops; }

private:
    core::Device mDevice;
    std::optional<double> mMemoryBandwidth;
    std::optional<double> mFp32Flops;
    std::optional<double> mFp16Flops;
    std::optional<double> mBf16Flops;
    std::optional<double> mFp8Flops;
};

}  // namespace device
}  // namespace external
}  // namespace dtorch
