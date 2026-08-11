/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "kernel.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>

#include "dtorch/common/config.h"

#if DTORCH_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/kernel/kernel_factory.h"
#include "dtorch/core/kernel_stream/cuda_kernel_stream.h"
#include "dtorch/core/kernel_stream/kernel_stream.h"
#include "dtorch/core/operators/operator_param.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

Kernel::Kernel(const KernelCreateCtx& ctx)
    : mOp(ctx.op),
      mInputs(),
      mOutputs(),
      mGlobalDevice(ctx.globalDevice),
      mLocalDevice(ctx.localDevice),
      mStream(ctx.stream),
      mNumKernelForThisOp(ctx.numKernelForThisOp),
      mTensorStoreCreateInfo(),
      mKernelHooks() {
    DAlwaysAssert(ctx.inputs.size() == mOp->GetInputSize());
    for (size_t i = 0; i < mOp->GetInputSize(); i++) {
        Operand* operand = mOp->GetInputOperand(i);
        // Skip GetLocalShape for non-distributed tensors: local shape == global shape
        Shape shape;
        if (operand->IsDistributed()) {
            shape = operand->GetLocalShape(GetGlobalDeviceId());
        } else {
            shape = operand->GetShape();
        }
        if (shape.IsNullTensorShape() || !operand->GetDeviceMesh().IsContainDevice(mGlobalDevice)) {
            mInputs.push_back(std::nullopt);
        } else {
            DDebugAssert(!ctx.inputs[i].empty());
            DDebugAssert(ctx.inputs[i].count(mGlobalDevice.deviceId));
            mInputs.push_back(ctx.inputs[i].at(mGlobalDevice.deviceId));
        }
    }

    DAlwaysAssert(ctx.outputs.size() == mOp->GetOutputSize());
    for (size_t i = 0; i < mOp->GetOutputSize(); i++) {
        Operand* operand = mOp->GetOutputOperand(i);
        if (!operand->GetDeviceMesh().IsContainDevice(mGlobalDevice)) {
            mOutputs.push_back(std::nullopt);
        } else {
            DDebugAssert(!ctx.outputs[i].empty());
            DDebugAssert(ctx.outputs[i].count(mGlobalDevice.deviceId));
            mOutputs.push_back(ctx.outputs[i].at(mGlobalDevice.deviceId));
        }
    }
}

void Kernel::Run() {
    DAlwaysAssert(mInputs.size() == mOp->GetInputSize());
    DAlwaysAssert(mOutputs.size() == mOp->GetOutputSize());

    for (auto& it : mKernelHooks) {
        it->BeforeCompute(*this);
    }

    // Ensure cuda stream is same as stream which is created in CudaKernelStream::InitInAsyncThread
    if (mLocalDevice.deviceKind == DeviceKind::kGpu) {
        TorchCudaStreamGuarantee::CheckStream(*(GetDeviceStream().GetTorchCudaStream()));
    }

    // Input and output for this mGlobalDevice and check input shape
    TorchTensorOptArray thisDeviceInputs;
    TorchTensorArray thisDeviceOutputs;
    for (size_t i = 0; i < mInputs.size(); i++) {
        Operand* operand = mOp->GetInputOperand(i);
        Shape shape = operand->GetLocalShape(GetGlobalDeviceId());
        if (shape.IsNullTensorShape() || !operand->GetDeviceMesh().IsContainDevice(mGlobalDevice)) {
            thisDeviceInputs.push_back(std::nullopt);
            DDebugAssert(!mInputs[i].has_value());
        } else {
            DDebugAssert(!mInputs[i].value().IsEmpty());

            torch::Tensor torchInput = *mInputs[i].value().GetTensor();
            if (GlobalOption::GetSingleton().GetValidateKernelInputOutput()) {
                // Validate input shape
                Shape torchShape = external::torch::TorchUtil::GetShape(torchInput);
                if (shape != torchShape) {
                    std::stringstream ss;
                    ss << "Input shape different: torch[" << torchShape << "], " << " vs dtorch[" << shape << "]";
                    DLogError() << ss.str();
                    DLogFatal() << "Describe string of operator: " << mOp->GetDescribeString();
                }

                // Validate input dtype
                DataKind dtorchDataKind = operand->GetDataKind();
                DataKind torchDataKind = external::torch::TorchUtil::GetDataKind(torchInput);
                if (dtorchDataKind != torchDataKind) {
                    std::stringstream ss;
                    ss << "Input data kind different: torch(" << torchDataKind << "), "
                       << " vs dtorch(" << dtorchDataKind << ")";
                    DLogError() << ss.str();
                    DLogFatal() << "Describe string of operator: " << mOp->GetDescribeString();
                }

                // Validate input device
                Device torchDevice = external::torch::TorchUtil::GetDevice(torchInput);
                if (torchDevice != mLocalDevice) {
                    std::stringstream ss;
                    ss << "Input device different: torch(" << torchDevice << "), "
                       << " vs dtorch(" << mLocalDevice << ")";
                    DLogError() << ss.str();
                    DLogFatal() << "Describe string of operator: " << mOp->GetDescribeString();
                }
            }

            thisDeviceInputs.push_back(torchInput);
        }
    }

    DAlwaysAssert(mInputs.size() == thisDeviceInputs.size());
    try {
        Compute(thisDeviceInputs, thisDeviceOutputs);
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Kernel::Compute() failed, operator type: " << mOp->GetOpType() << ", error message: " << e.what();
        DLogError() << ss.str();
        DLogError() << "Describe string of operator: " << mOp->GetDescribeString();
        throw e;
    }

    // Ensure cuda stream is same as stream which is created in CudaKernelStream::InitInAsyncThread
    if (mLocalDevice.deviceKind == DeviceKind::kGpu) {
        TorchCudaStreamGuarantee::CheckStream(*(GetDeviceStream().GetTorchCudaStream()));
    }

    // Set output and check output shape
    for (size_t i = 0; i < mOutputs.size(); i++) {
        Operand* operand = mOp->GetOutputOperand(i);
        if (!operand->GetDeviceMesh().IsContainDevice(mGlobalDevice)) {
            DDebugAssert(thisDeviceOutputs.size() == 0);
            continue;
        }

        // output operand MUST have same device mesh.
        DDebugAssert(mOutputs.size() == thisDeviceOutputs.size());

        if (GlobalOption::GetSingleton().GetValidateKernelInputOutput()) {
            // Validate output shape
            Shape shape = operand->GetLocalShape(GetGlobalDeviceId());
            Shape torchShape = external::torch::TorchUtil::GetShape(thisDeviceOutputs[i]);
            if (shape != torchShape) {
                std::stringstream ss;
                ss << "Output shape different: torch[" << torchShape << "], " << " vs dtorch[" << shape << "]";
                DLogError() << ss.str();
                DLogFatal() << "Describe string of operator: " << mOp->GetDescribeString();
                DUnsupportedImpl();
            }

            // Validate output dtype
            DataKind dtorchDataKind = operand->GetDataKind();
            DataKind torchDataKind = external::torch::TorchUtil::GetDataKind(thisDeviceOutputs[i]);
            if (dtorchDataKind != torchDataKind) {
                std::stringstream ss;
                ss << "Output data kind different: torch(" << torchDataKind << "), "
                   << " vs dtorch(" << dtorchDataKind << ")";
                DLogError() << ss.str();
                DLogFatal() << "Describe string of operator: " << mOp->GetDescribeString();
            }

            // Validate output device
            Device torchDevice = external::torch::TorchUtil::GetDevice(thisDeviceOutputs[i]);
            if (torchDevice != mLocalDevice) {
                std::stringstream ss;
                ss << "Output device different: torch(" << torchDevice << "), "
                   << " vs dtorch(" << mLocalDevice << ")";
                DLogError() << ss.str();
                DLogFatal() << "Describe string of operator: " << mOp->GetDescribeString();
            }
        }

        DDebugAssert(mOutputs[i].has_value());
        DDebugAssert(mOutputs[i].value().IsEmpty());
        mOutputs[i].value().SetTensor(std::make_shared<torch::Tensor>(thisDeviceOutputs[i]));
    }

    for (auto& it : mKernelHooks) {
        it->AfterCompute(*this);
    }
}

void Kernel::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    DAlwaysAssert(mOp != nullptr);

    // Calls mOp->Compute to perform kernel computation for the operator.
    // Typically, all input/output operands share the same device mesh.
    if (mOp->GetInputSize() + mOp->GetOutputSize() > 0) {
        Operand* operand = nullptr;
        if (mOp->GetOutputSize() > 0) {
            operand = mOp->OperandY();
        } else {
            DAlwaysAssert(mOp->GetInputSize() > 0);
            operand = mOp->OperandX();
        }
        DAlwaysAssert(mNumKernelForThisOp == operand->GetDeviceMesh().Count());
    }

    mOp->Compute(inputs, outputs);
}

const Device& Kernel::GetLocalDevice() { return mStream->GetDevice(); }

}  // namespace core
}  // namespace dtorch
