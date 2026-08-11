/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstring>

#include "cuda_error.h"
#include "dtorch/common/debug.h"

namespace dtorch {
namespace external {
namespace cuda {

class CudaDeviceProp : public cudaDeviceProp {
public:
    CudaDeviceProp() { std::memset(Get(), 0, sizeof(cudaDeviceProp)); }

    DTORCH_FORCEINLINE cudaDeviceProp* Get() { return this; }

    DTORCH_FORCEINLINE const cudaDeviceProp* Get() const { return this; }
};

class CudaDevice {
public:
    DTORCH_FORCEINLINE static int GetDeviceCount() {
        int count;
        CudaCheckError(cudaGetDeviceCount(&count));
        return count;
    }

    DTORCH_FORCEINLINE static int GetCurrentDeviceId() {
        int device = 0;
        CudaCheckError(cudaGetDevice(&device));
        return device;
    }

    DTORCH_FORCEINLINE static bool DeviceIdIsValid(int deviceId) {
        return deviceId >= 0 && deviceId < GetDeviceCount();
    }

    DTORCH_FORCEINLINE static void SetDevice(int device) {
        DDebugAssert(DeviceIdIsValid(device));
        CudaCheckError(cudaSetDevice(device));
    }

    DTORCH_FORCEINLINE static int GetDevice() {
        int device = 0;
        CudaCheckError(cudaGetDevice(&device));
        return device;
    }

    DTORCH_FORCEINLINE static int ChooseDevice(const CudaDeviceProp& prop) {
        int device = 0;
        CudaCheckError(cudaChooseDevice(&device, prop.Get()));
        return device;
    }

    DTORCH_FORCEINLINE static int GetAttribute(int deviceId, cudaDeviceAttr attr) {
        int value;
        CudaCheckError(cudaDeviceGetAttribute(&value, attr, deviceId));
        return value;
    }

    DTORCH_FORCEINLINE static void GetDeviceProperties(int deviceId, CudaDeviceProp& prop) {
        CudaCheckError(cudaGetDeviceProperties(&prop, deviceId));
    }

    DTORCH_FORCEINLINE static void GetMemInfo(size_t& free, size_t& total) {
        CudaCheckError(cudaMemGetInfo(&free, &total));
    }

    DTORCH_FORCEINLINE static void GetDeviceCapability(int deviceId, int& major, int& minor) {
        CudaDeviceProp prop;
        GetDeviceProperties(deviceId, prop);
        major = prop.major;
        minor = prop.minor;
    }

    DTORCH_FORCEINLINE static bool IsSupportIpc(int deviceId) {
        CudaDeviceProp prop;
        GetDeviceProperties(deviceId, prop);
        return prop.unifiedAddressing && prop.computeMode == cudaComputeModeDefault;
    }
};

class CudaDeviceGuard {
public:
    CudaDeviceGuard(int deviceId) : mPriorDeviceId(0) {
        DAlwaysAssert(CudaDevice::DeviceIdIsValid(deviceId));
        mPriorDeviceId = CudaDevice::GetCurrentDeviceId();
        CudaDevice::SetDevice(deviceId);
    }

    ~CudaDeviceGuard() { CudaDevice::SetDevice(mPriorDeviceId); }

private:
    int mPriorDeviceId;
};

}  // namespace cuda
}  // namespace external
}  // namespace dtorch
