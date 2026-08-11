/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <mutex>

#include "dtorch/common/debug.h"
#include "dtorch/core/blob.h"
#include "dtorch/core/communication/tensor_store/tensor_store.h"
#include "dtorch/core/communication/thread_group/thread_group_manager.h"
#include "dtorch/core/kernel/kernel_factory.h"
#include "dtorch/core/kernel/kernel_hook/kernel_hook.h"
#include "dtorch/core/operators/operator.h"

namespace dtorch {
namespace core {

class KernelStream;

class Kernel {
public:
    Kernel(const KernelCreateCtx& ctx);

    virtual ~Kernel() = default;

    DTORCH_DEFAULT_COPY_AND_MOVE(Kernel);

    void Run();

    DTORCH_FORCEINLINE const Operator* GetOperator() const noexcept { return mOp.get(); }

    DTORCH_FORCEINLINE void AddKernelHook(std::shared_ptr<KernelHook> hook) { mKernelHooks.push_back(std::move(hook)); }

    DTORCH_FORCEINLINE const Device& GetGlobalDevice() const noexcept { return mGlobalDevice; }

    DTORCH_FORCEINLINE int64_t GetGlobalDeviceId() const noexcept { return mGlobalDevice.deviceId; }

    DTORCH_FORCEINLINE bool GlobalDeviceInOperand(const Operand* operand) const {
        return operand->GetDeviceMesh().IsContainDevice(mGlobalDevice);
    }

    const Device& GetLocalDevice();

    KernelStream& GetStream() const { return *mStream; };

    DTORCH_FORCEINLINE DeviceStream GetDeviceStream() const { return DeviceStream::GetCurrentStream(mLocalDevice); }

    DTORCH_FORCEINLINE std::shared_ptr<communication::TensorStore> GetTensorStore() const {
        DDebugAssert(mTensorStoreCreateInfo != nullptr);
        return communication::TensorStore::Create(*mTensorStoreCreateInfo);
    }

    DTORCH_FORCEINLINE void SetTensorStoreCreateInfo(
        const std::shared_ptr<communication::TensorStoreCreateInfo>& createInfo) {
        DDebugAssert(createInfo != nullptr);
        mTensorStoreCreateInfo = createInfo;
    }

protected:
    virtual void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs);

    template <typename DerivedOpParam>
    DTORCH_FORCEINLINE const DerivedOpParam& GetOpParam() const {
        return *(DerivedCast<DerivedOpParam, OpParam>(GetOperator()->GetOpParam()));
    }

protected:
    std::shared_ptr<Operator> mOp;
    std::vector<std::optional<Blob>> mInputs;
    std::vector<std::optional<Blob>> mOutputs;
    Device mGlobalDevice;
    Device mLocalDevice;
    KernelStream* mStream;
    size_t mNumKernelForThisOp;
    std::shared_ptr<communication::TensorStoreCreateInfo> mTensorStoreCreateInfo;
    std::vector<std::shared_ptr<KernelHook>> mKernelHooks;
};

}  // namespace core
}  // namespace dtorch
