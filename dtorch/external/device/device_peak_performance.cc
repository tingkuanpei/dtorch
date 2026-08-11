/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "device_peak_performance.h"

#include <stdexcept>

#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"

#if DTORCH_WITH_CUDA
#include "dtorch/external/cuda/cuda_device.h"
#endif

namespace dtorch {
namespace external {
namespace device {

#if DTORCH_WITH_CUDA

namespace {

// CUDA cores (FP32 lanes) per streaming multiprocessor, by compute capability.
// Source: "Compute Capabilities" table, CUDA C++ Programming Guide.
int CudaCoresPerSm(int major, int minor) {
    switch (major * 10 + minor) {
        case 10:
        case 11:
        case 12:
        case 13:
            return 8;  // Tesla
        case 20:
            return 32;  // Fermi GF100
        case 21:
            return 48;  // Fermi GF110
        case 30:
        case 32:
        case 35:
        case 37:
            return 192;  // Kepler
        case 50:
        case 52:
        case 53:
            return 128;  // Maxwell
        case 60:
            return 64;  // Pascal GP100
        case 61:
        case 62:
            return 128;  // Pascal consumer / GP102
        case 70:
        case 72:
            return 64;  // Volta
        case 75:
            return 64;  // Turing
        case 80:
            return 64;  // Ampere A100 (GA100)
        case 86:
        case 89:
            return 128;  // Ampere consumer (GA10x) / Ada Lovelace
        case 90:
            return 128;  // Hopper
        case 100:
        case 101:
            return 128;  // Blackwell datacenter
        default:
            DLogWarning() << "Unknown compute capability " << major << "." << minor
                          << ", assuming 128 CUDA cores per SM";
            return 128;
    }
}

// FP16 throughput multiplier relative to FP32 on CUDA cores.
// Returns nullopt when the device cannot execute FP16 arithmetic natively.
// Source: "Compute Capabilities" / "Arithmetic Instructions" tables, CUDA C++ Programming Guide.
std::optional<double> Fp16ThroughputMultiplier(int major, int minor) {
    switch (major * 10 + minor) {
        case 60:
            return 2.0;  // Pascal GP100: packed FP16 (FP16x2)
        case 61:
        case 62:
            return 1.0;  // Pascal consumer: FP16 at FP32 rate (no packed FP16 HW)
        case 70:
        case 72:
            return 2.0;  // Volta
        case 75:
            return 2.0;  // Turing
        case 80:
        case 86:
        case 89:
            return 2.0;  // Ampere / Ada
        case 90:
            return 2.0;  // Hopper
        case 100:
        case 101:
            return 2.0;  // Blackwell
        default:
            return std::nullopt;  // FP16 not supported in CUDA cores
    }
}

// BF16 throughput multiplier relative to FP32 on CUDA cores.
// BF16 compute on CUDA cores requires SM ≥ 8.0 (Ampere+).
std::optional<double> Bf16ThroughputMultiplier(int major, int /*minor*/) {
    if (major >= 8) {
        return 2.0;  // Same throughput as FP16
    }
    return std::nullopt;
}

// FP8 throughput multiplier relative to FP32 on CUDA cores.
// FP8 (E4M3 / E5M2) compute requires SM ≥ 8.9 (Ada / Hopper / Blackwell).
// FP8 has 2× the throughput of FP16 on these architectures.
std::optional<double> Fp8ThroughputMultiplier(int major, int minor) {
    if (major > 8 || (major == 8 && minor >= 9)) {
        return 4.0;  // 2× FP16 = 4× FP32
    }
    return std::nullopt;
}

// Fills all peak fields from a cudaDeviceProp snapshot.
void ComputeGpuPeaks(const external::cuda::CudaDeviceProp& prop, std::optional<double>& outBandwidth,
                     std::optional<double>& outFp32, std::optional<double>& outFp16, std::optional<double>& outBf16,
                     std::optional<double>& outFp8) {
    // ---- memory bandwidth ----
    const int memClockRate = prop.Get()->memoryClockRate;
    const int memBusWidth = prop.Get()->memoryBusWidth;
    if (memClockRate > 0 && memBusWidth > 0) {
        // 2.0 × memClockRate(kHz) × 1000.0 × (memBusWidth(bits) / 8.0) → bytes/sec
        outBandwidth = 2.0 * static_cast<double>(memClockRate) * 1000.0 * (static_cast<double>(memBusWidth) / 8.0);
    }

    // ---- compute peaks ----
    const int smCount = prop.Get()->multiProcessorCount;
    const int clockRate = prop.Get()->clockRate;
    if (smCount <= 0 || clockRate <= 0) {
        return;
    }

    const int major = prop.Get()->major;
    const int minor = prop.Get()->minor;

    // FP32: SMs × coresPerSM × clockRate(kHz) × 1000 × 2 (FMA)
    outFp32 = static_cast<double>(smCount) * static_cast<double>(CudaCoresPerSm(major, minor)) *
              static_cast<double>(clockRate) * 1000.0 * 2.0;

    // FP16
    auto fp16Mult = Fp16ThroughputMultiplier(major, minor);
    if (fp16Mult.has_value()) {
        outFp16 = *outFp32 * *fp16Mult;
    }

    // BF16 (Ampere+)
    auto bf16Mult = Bf16ThroughputMultiplier(major, minor);
    if (bf16Mult.has_value()) {
        outBf16 = *outFp32 * *bf16Mult;
    }

    // FP8 (Ada / Hopper / Blackwell)
    auto fp8Mult = Fp8ThroughputMultiplier(major, minor);
    if (fp8Mult.has_value()) {
        outFp8 = *outFp32 * *fp8Mult;
    }
}

}  // namespace

#endif  // DTORCH_WITH_CUDA

DevicePeakPerformance::DevicePeakPerformance(const core::Device& device)
    : mDevice(device),
      mMemoryBandwidth(std::nullopt),
      mFp32Flops(std::nullopt),
      mFp16Flops(std::nullopt),
      mBf16Flops(std::nullopt),
      mFp8Flops(std::nullopt) {
    if (device.deviceKind == core::DeviceKind::kCpu) {
        // TODO: Retrieve CPU peaks from DRAM specs / cache topology / cpuid.
        // Not yet implemented: leave all peak fields as nullopt (documented behavior).
        DLogWarning() << "DevicePeakPerformance: CPU peak performance is not yet implemented, "
                         "all getters will return std::nullopt";
        return;
    }
#if DTORCH_WITH_CUDA
    else if (device.deviceKind == core::DeviceKind::kGpu) {
        DAlwaysAssert(external::cuda::CudaDevice::DeviceIdIsValid(device.deviceId));
        external::cuda::CudaDeviceProp prop;
        // cudaGetDeviceProperties queries by device ID — no CudaDeviceGuard needed.
        external::cuda::CudaDevice::GetDeviceProperties(device.deviceId, prop);
        ComputeGpuPeaks(prop, mMemoryBandwidth, mFp32Flops, mFp16Flops, mBf16Flops, mFp8Flops);
        return;
    }
#endif  // DTORCH_WITH_CUDA
    else {
        throw std::invalid_argument("Unsupport device: " + core::DeviceKindToString(device.deviceKind));
    }
}

}  // namespace device
}  // namespace external
}  // namespace dtorch
