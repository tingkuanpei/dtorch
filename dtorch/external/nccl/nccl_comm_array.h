/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <vector>

#include "dtorch/external/cuda/cuda_device.h"
#include "dtorch/external/cuda/cuda_object.h"
#include "nccl_error.h"

namespace dtorch {
namespace external {
namespace nccl {

template <typename T>
using NcclObject = device::CudaObject<T>;

class NcclCommArray : public NcclObject<ncclComm_t*> {
public:
    DTORCH_FORCEINLINE static void GroupStart() { NcclCheckError(ncclGroupStart()); }

    DTORCH_FORCEINLINE static void GroupEnd() { NcclCheckError(ncclGroupEnd()); }

public:
    NcclCommArray() : mDeviceSize(0) {}

    DTORCH_FORCEINLINE void InitAll(const std::vector<int>& deviceIdArray) {
        size_t deviceSize = deviceIdArray.size();
        DDebugAssert(deviceSize > 0);

        for (size_t i = 0; i < deviceSize; i++) {
            DDebugAssert(deviceIdArray[i] >= 0 && deviceIdArray[i] < device::CudaDevice::GetDeviceCount());
        }

        mDeviceSize = deviceSize;

        ncclComm_t* ncclComm = new ncclComm_t[deviceSize];
        NcclCheckError(ncclCommInitAll(ncclComm, deviceSize, deviceIdArray.data()));

        Reset(ncclComm, [&](ncclComm_t* p) {
            for (size_t i = 0; i < mDeviceSize; i++) {
                NcclCheckError(ncclCommDestroy(p[i]));
            }

            delete[] p;
        });
    }

    DTORCH_FORCEINLINE size_t Size() const noexcept { return mDeviceSize; }

    DTORCH_FORCEINLINE ncclComm_t operator[](int index) const noexcept {
        DDebugAssert(index >= 0 && static_cast<size_t>(index) < mDeviceSize);
        return Get()[index];
    }
    DTORCH_FORCEINLINE ncclComm_t operator[](int index) noexcept {
        return const_cast<ncclComm_t>(static_cast<const NcclCommArray&>(*this)[index]);
    }

private:
    size_t mDeviceSize;
};

}  // namespace nccl
}  // namespace external
}  // namespace dtorch
