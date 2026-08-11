/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "convert_kernel.h"

#include <cstdint>
#include <memory>
#include <mutex>

#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/torch.h>

#include "distributed_scather_and_gather.h"
#include "dtorch/common/config.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/communication/thread_group/thread_group.h"
#include "dtorch/core/communication/thread_group/thread_group_manager.h"
#include "dtorch/core/kernel_stream/cuda_kernel_stream.h"
#include "dtorch/core/kernel_stream/kernel_stream.h"
#include "dtorch/core/operators/standard/convert_op.h"
#include "dtorch/external/cuda/nvtx_profiler.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

void ConvertKernel::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    DDebugAssert(inputs.size() == 1);
    if (mLocalDevice.deviceKind == DeviceKind::kGpu) {
        TorchCudaStreamGuarantee::CheckStream(*(GetDeviceStream().GetTorchCudaStream()));
    }

    bool inputIsDistributed = GetSrcDeviceMesh().IsDistributed();
    bool outputIsDistributed = GetDestDeviceMesh().IsDistributed();
    bool convertDataKind = GetDestDataKind() != GetSrcDataKind();
    bool convertDeviceMesh = GetDestDeviceMesh() != GetSrcDeviceMesh();
    bool convertPlacement = GetDestPlacementSeq() != GetSrcPlacementSeq();

    if (!inputIsDistributed && !outputIsDistributed && !convertDataKind && mNumKernelForThisOp == 1) {
        // device_mesh shape from [1] to [1, 1, 1]
        DDebugAssert(inputs[0].has_value());
        outputs.push_back(inputs[0].value());
    } else if (!inputIsDistributed && outputIsDistributed && !convertDataKind) {
        ScatherTensor(inputs, outputs);
    } else if (inputIsDistributed && !outputIsDistributed && !convertDataKind) {
        GatherTensor(inputs, outputs);
    } else if (!inputIsDistributed && !outputIsDistributed) {
        ConvertDeviceAndDataKind(inputs, outputs);
    } else if (inputIsDistributed && convertDataKind && !convertDeviceMesh && !convertPlacement) {
        ConvertDeviceAndDataKind(inputs, outputs);
    } else if (inputIsDistributed && outputIsDistributed && !convertDeviceMesh && convertPlacement &&
               !convertDataKind) {
        ConvertPlacements(inputs, outputs);
    } else if (inputIsDistributed && outputIsDistributed && convertDeviceMesh && !convertPlacement &&
               !convertDataKind) {
        ConvertDeviceMesh(inputs, outputs);
    } else {
        DLogError() << "convertDataKind: " << convertDataKind;
        DLogError() << "convertDeviceMesh: " << convertDeviceMesh;
        DLogError() << "convertPlacement: " << convertPlacement;
        DLogError() << "inputIsDistributed: " << inputIsDistributed;
        DLogError() << "outputIsDistributed: " << outputIsDistributed;
        DUnimplemented();
    }

    if (GlobalDeviceInOperand(mOp->OperandY())) {
        DDebugAssertMsg(outputs.size() == 1, "Operator describe string: " + mOp->GetDescribeString());
    } else {
        DDebugAssertMsg(outputs.size() == 0, "Operator describe string: " + mOp->GetDescribeString());
    }
}

void ConvertKernel::ConvertDeviceMesh(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    DeviceStream deviceStream = GetDeviceStream();
    auto tensorStore = GetTensorStore();

    if (GlobalDeviceInOperand(mOp->OperandX())) {
        DDebugAssert(inputs[0].has_value());
        std::string key = std::to_string(mGlobalDevice.deviceId);
        DeviceKind destGetDeviceKind = mOp->OperandY()->GetDeviceKind();
        tensorStore->SrcSet(key, inputs[0].value(), deviceStream, 1, destGetDeviceKind);
    }

    if (GlobalDeviceInOperand(mOp->OperandY())) {
        size_t rank = DistributedSpec::GetRankId(mGlobalDevice.deviceId, GetDestDeviceMesh().GetMesh().GetData());
        int64_t inputGlobalDeviceId = GetSrcDeviceMesh().GetMesh().GetData()[rank];
        std::string key = std::to_string(inputGlobalDeviceId);
        torch::Tensor output = tensorStore->DestGetAndToDevice(key, deviceStream, mLocalDevice);
        outputs.push_back(output);
    }

    if (GlobalDeviceInOperand(mOp->OperandX())) {
        std::string key = std::to_string(mGlobalDevice.deviceId);
        tensorStore->SrcWaitUntilGetFinished(key, deviceStream);
    }
}

void ConvertKernel::ConvertPlacements(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    DAlwaysAssert(mNumKernelForThisOp == GetSrcDeviceMesh().Count());
    DDebugAssert(inputs.size() == 1);

    auto srcPlacementSeq = GetSrcPlacementSeq();
    auto destPlacementSeq = GetDestPlacementSeq();
    auto srcDeviceMesh = GetSrcDeviceMesh();
    auto destDeviceMesh = GetDestDeviceMesh();

    DeviceKind deviceKind = srcDeviceMesh.GetDeviceKind();
    auto diffDims = srcPlacementSeq.GetDiffDims(destPlacementSeq);
    DDebugAssert(diffDims.size() == 1);
    size_t diffDim = diffDims[0];

    communication::ThreadGroupInfo info(srcDeviceMesh.GetMesh(), diffDim, mGlobalDevice.deviceId);
    communication::ThreadGroup& threadGroup =
        mThreadGroupManager->GetThreadGroup(deviceKind, info.GetAllDeviceIds(), mGlobalDevice.deviceId);
    DeviceStream deviceStream = GetDeviceStream();
    threadGroup.SetStream(deviceStream);

    // ------------------------------------------
    // |  Src  |  Dest  |       Op       |
    // ------------------------------------------
    // |   Si  |   Sj   |     all2all    |  done
    // |   S   |   R    |   all-gather   |  done
    // |   S   |   P    |       no       |
    // |   R   |   S    |       no       |  done
    // |   R   |   R    |       no       |   no
    // |   R   |   P    |       no       |
    // |   P   |   S    | resuce-scather |  done
    // |   P   |   R    |   all-reduce   |  done
    // |   P   |   P    |       no       |   no

    torch::Tensor input = inputs[0].value();
    const Operand& inputOperand = *(mOp->OperandX());
    torch::Tensor output;

    // TODO:
    // https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2243/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently
    // In a multi-GPU environment, removing threadGroup.Barrier() will cause the program to hang. By printing debug
    // information, we found that all CPU threads do not execute the "allreduce" function simultaneously.
    // We suspect this is the reason for the program hang, so we temporarily added Barrier for synchronization.
    threadGroup.Barrier();

    if (srcPlacementSeq[diffDim].IsShard() && destPlacementSeq[diffDim].IsShard()) {
        DAlwaysAssert(!srcPlacementSeq[diffDim].HasSubSplitCoordinates());
        DAlwaysAssert(!destPlacementSeq[diffDim].HasSubSplitCoordinates());
        // S -> S
        size_t srcDim = static_cast<size_t>(srcPlacementSeq[diffDim].GetShardIndex());
        size_t destDim = static_cast<size_t>(destPlacementSeq[diffDim].GetShardIndex());
        output = threadGroup.AllToAll(input, srcDim, destDim, inputOperand);
    } else if (srcPlacementSeq[diffDim].IsShard() && destPlacementSeq[diffDim].IsReplicate()) {
        if (!srcPlacementSeq[diffDim].HasSubSplitCoordinates()) {
            // S -> R
            int64_t shardIndex = srcPlacementSeq[diffDim].GetShardIndex();
            if (shardIndex == 0) {
                output = threadGroup.AllGatherIntoTensor(input, inputOperand);
            } else {
                output = threadGroup.AllGather(input, inputOperand, shardIndex);
            }
        } else {
            auto tmpTensors = threadGroup.AllGatherIntoVec(input, inputOperand);
            output = DistributeGather::GatherShardWithSubSplit(tmpTensors, srcDeviceMesh, srcPlacementSeq, diffDim);
        }
    } else if (srcPlacementSeq[diffDim].IsReplicate() && destPlacementSeq[diffDim].IsShard()) {
        // R -> S
        int64_t shardIdx = destPlacementSeq[diffDim].GetShardIndex();
        int64_t subSplit = destPlacementSeq[diffDim].GetSubSplitCoordinates();
        output = threadGroup.ReplicateToShard(input, shardIdx, subSplit);
    } else if (srcPlacementSeq[diffDim].IsPartial() && destPlacementSeq[diffDim].IsShard(0)) {
        // P -> S
        output = threadGroup.ReduceScatterTensor(input);
    } else if (srcPlacementSeq[diffDim].IsPartial() && destPlacementSeq[diffDim].IsReplicate()) {
        // P -> R
        output = threadGroup.AllReduce(input);
    } else {
        DLogError() << "Input: " << srcPlacementSeq << ", Output: " << destPlacementSeq;
        DUnimplemented();
    }

    outputs.push_back(output);
}

void ConvertKernel::ConvertDeviceAndDataKind(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    const DeviceMesh& srcMesh = GetSrcDeviceMesh();
    const DeviceMesh& destMesh = GetDestDeviceMesh();

    if (srcMesh.IsDistributed()) {
        DAlwaysAssert(srcMesh == destMesh);
    }

    torch::Tensor input;
    if (srcMesh != destMesh) {
        DAlwaysAssert(!srcMesh.IsDistributed());
        auto tensorStore = GetTensorStore();
        DeviceStream deviceStream = GetDeviceStream();
        const std::string key = "tensor";

        if (GlobalDeviceInOperand(mOp->OperandX())) {
            DDebugAssert(inputs[0].has_value());
            DeviceKind destGetDeviceKind = mOp->OperandY()->GetDeviceKind();
            tensorStore->SrcSet(key, inputs[0].value(), deviceStream, 1, destGetDeviceKind);
            tensorStore->SrcWaitUntilGetFinished(key, deviceStream);
            return;
        }

        if (GlobalDeviceInOperand(mOp->OperandY())) {
            input = tensorStore->DestGetAndToDevice(key, deviceStream, mLocalDevice);
        }
    } else {
        DDebugAssert(inputs[0].has_value());
        input = inputs[0].value();
    }

    DDebugAssert(inputs.size() == 1);
    torch::ScalarType scalarType = external::torch::TorchUtil::ToScalarType(GetDestDataKind());
    auto options = torch::TensorOptions().dtype(scalarType);
    outputs.push_back(input.to(options));
}

void ConvertKernel::ScatherTensor(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    DAlwaysAssert(mNumKernelForThisOp == (GetDestDeviceMesh().Count() + 1) ||
                  mNumKernelForThisOp == GetDestDeviceMesh().Count());
    DDebugAssert(inputs.size() == 1);
    auto tensorStore = GetTensorStore();
    DeviceStream deviceStream = GetDeviceStream();

    // void* handle = nullptr;
    // if (deviceStream.GetDeviceKind() == DeviceKind::kGpu) {
    //     handle = external::cuda::NvtxProfile::StreamRangePush(*(deviceStream.GetTorchCudaStream()),
    //                                                           "ConvertKernel::ScatherTensor");
    // }

    const auto& operandYDeviceMesh = mOp->OperandY()->GetDeviceMesh();
    const auto& operandYDeviceIdSet = operandYDeviceMesh.GetDeviceIdSet();

    if (GlobalDeviceInOperand(mOp->OperandX())) {
        DDebugAssert(inputs[0].has_value());
        DistributedScather distributedScather(operandYDeviceMesh, mOp->OperandY()->GetPlacementSeq(),
                                              inputs[0].value());
        for (auto globalDeviceId : operandYDeviceIdSet) {
            torch::Tensor tensor = distributedScather.Get(globalDeviceId);
            DAlwaysAssert(external::torch::TorchUtil::GetShape(tensor) ==
                          mOp->OperandY()->GetLocalShape(globalDeviceId));
            DeviceKind destGetDeviceKind = mOp->OperandY()->GetDeviceKind();
            tensorStore->SrcSet(std::to_string(globalDeviceId), tensor, deviceStream, 1, destGetDeviceKind);
        }
    }

    if (GlobalDeviceInOperand(mOp->OperandY())) {
        DDebugAssert(GlobalDeviceInOperand(mOp->OperandY()));

        DDebugAssert(operandYDeviceIdSet.count(mGlobalDevice.deviceId) > 0);
        torch::Tensor output =
            tensorStore->DestGetAndToDevice(std::to_string(mGlobalDevice.deviceId), deviceStream, mLocalDevice);
        outputs.push_back(output);
    }

    if (GlobalDeviceInOperand(mOp->OperandX())) {
        for (auto globalDeviceId : operandYDeviceIdSet) {
            tensorStore->SrcWaitUntilGetFinished(std::to_string(globalDeviceId), deviceStream);
        }
    }
    // if (handle != nullptr) {
    //     external::cuda::NvtxProfile::StreamRangePop(handle, *(deviceStream.GetTorchCudaStream()));
    // }
}

void ConvertKernel::GatherTensor(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) {
    DAlwaysAssert(mNumKernelForThisOp == (GetSrcDeviceMesh().Count() + 1) ||
                  mNumKernelForThisOp == GetSrcDeviceMesh().Count());
    DDebugAssert(inputs.size() == 1);
    const auto& operandXDeviceMesh = GetSrcDeviceMesh();
    const auto& operandXDeviceIds = operandXDeviceMesh.GetMesh().GetData();

    auto tensorStore = GetTensorStore();
    DeviceStream deviceStream = GetDeviceStream();

    // void* handle = nullptr;
    // if (deviceStream.GetDeviceKind() == DeviceKind::kGpu) {
    //     handle = external::cuda::NvtxProfile::StreamRangePush(*(deviceStream.GetTorchCudaStream()),
    //                                                           "ConvertKernel::GatherTensor");
    // }

    if (GlobalDeviceInOperand(mOp->OperandX())) {
        DeviceKind destGetDeviceKind = mOp->OperandY()->GetDeviceKind();
        tensorStore->SrcSet(std::to_string(mGlobalDevice.deviceId), inputs[0].value(), deviceStream, 1,
                            destGetDeviceKind);
    }

    if (GlobalDeviceInOperand(mOp->OperandY())) {
        DDebugAssert(GlobalDeviceInOperand(mOp->OperandY()));

        std::unordered_map<int64_t, std::shared_ptr<torch::Tensor>> allRankTensorsInSameDevice;
        for (auto globalDeviceId : operandXDeviceIds) {
            torch::Tensor tensor =
                tensorStore->DestGetAndToDevice(std::to_string(globalDeviceId), deviceStream, mLocalDevice);
            allRankTensorsInSameDevice[globalDeviceId] = std::make_shared<torch::Tensor>(tensor);
        }

        DistributeGather distributeGather(operandXDeviceMesh, mOp->OperandX()->GetPlacementSeq(),
                                          allRankTensorsInSameDevice);
        outputs.push_back(distributeGather.Get());
    }

    if (GlobalDeviceInOperand(mOp->OperandX())) {
        tensorStore->SrcWaitUntilGetFinished(std::to_string(mGlobalDevice.deviceId), deviceStream);
    }
    // if (handle != nullptr) {
    //     external::cuda::NvtxProfile::StreamRangePop(handle, *(deviceStream.GetTorchCudaStream()));
    // }
}

}  // namespace core
}  // namespace dtorch
