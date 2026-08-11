/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "operator.h"

#include <torch/torch.h>

#include "dtorch/api/cpp/device.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/distributed/cluster_info.h"
#include "dtorch/core/operators/operator_param.h"
#include "dtorch/core/operators/operator_serialization_pack.h"
#include "standard/get_item_op.h"

namespace dtorch {
namespace core {

Operator::Operator(std::shared_ptr<OpParam> opParamPtr)
    : mOpName(""),
      mUniqueId(OperatorIdManager::kNoValue),
      mOpType(opParamPtr->GetOpType()),
      mOpParam(opParamPtr),
      mInputOperands(),
      mOutputOperands(),
      mOperatorAssignInfo() {}

//---------------------------------------------- Check Input -----------------------------------------------------------

void Operator::CheckInput() const {
    CheckInputSameDataKind();
    CheckInputAllDistributedOrNot();
    CheckInputSameDeviceMesh();
    CheckInputDistributedSpec();
}

void Operator::CheckInputSameDataKind() const {
    if (GetInputSize() < 1 || !IsRequireInputSameDataKind()) {
        return;
    }

    DataKind expectDataKind = GetInputOperand(0)->GetDataKind();
    for (size_t i = 0; i < GetInputSize(); i++) {
        if (GetInputOperand(i)->IsNullTensorShape()) {
            continue;
        }

        DataKind dataKind = GetInputOperand(i)->GetDataKind();
        if (dataKind != expectDataKind) {
            std::stringstream ss;
            ss << "Expected all tensors have same data kind, but got " << expectDataKind << " vs " << dataKind;
            throw std::invalid_argument(ss.str());
        }
    }
}

void Operator::CheckInputAllDistributedOrNot() const {
    if (GetInputSize() < 1) {
        return;
    }

    bool isDistributed = GetInputOperand(0)->IsDistributed();
    for (size_t i = 0; i < GetInputSize(); i++) {
        if (GetInputOperand(i)->IsNullTensorShape()) {
            continue;
        }

        if (GetInputOperand(i)->IsDistributed() != isDistributed) {
            throw std::invalid_argument("Input must all be local tensor or distribute tensor");
        }
    }
}

void Operator::CheckInputSameDeviceMesh() const {
    if (GetInputSize() < 1) {
        return;
    }

    DeviceMesh expectDeviceMesh = GetInputOperand(0)->GetDeviceMesh();
    for (size_t i = 0; i < GetInputSize(); i++) {
        if (GetInputOperand(i)->IsNullTensorShape()) {
            continue;
        }

        const auto& deviceMesh = GetInputOperand(i)->GetDeviceMesh();
        if (expectDeviceMesh != deviceMesh) {
            std::stringstream ss;
            ss << "Expected all tensors have same device mesh, but got " << expectDeviceMesh << " vs "
               << deviceMesh.ToString();
            throw std::invalid_argument(ss.str());
        }
    }
}

void Operator::CheckInputDistributedSpec() const {
    // ClusterInfo is a global singleton populated on every process role: the MainNode fills it
    // during cluster formation; WorkerNode processes and RemoteRunner subprocesses receive the
    // serialized form via CreateGraph / the subprocess launch payload. So the true cluster-wide
    // total GPU count is available uniformly here (no IsCreate()/fallback branching needed).
    int64_t totalGpuCount = core::distributed::ClusterInfo::GetSingleton().GetTotalGpuCount();

    for (size_t i = 0; i < GetInputSize(); i++) {
        Operand* operand = GetInputOperand(i);
        const auto& shape = operand->GetShape();
        const auto& deviceMesh = operand->GetDeviceMesh();
        const auto& placements = operand->GetPlacementSeq();
        if (!DistributedSpec::CheckValid(shape, deviceMesh, placements, totalGpuCount)) {
            std::stringstream ss;
            ss << "Shape, mesh and placements is invalid for operator input, shape: " << shape
               << ", mesh: " << deviceMesh << ", placements: " << placements;
            throw std::invalid_argument(ss.str());
        }
    }
}

//--------------------------------------------------- Infer-------------------------------------------------------------

void Operator::Infer() {
    InferOutput();
    InferOperatorAssignInfo();
}

void Operator::InferOutput() {
    if (GetOutputSize() != InferOutputSize()) {
        std::stringstream ss;
        ss << "Operator output size not valid: " << GetOutputSize() << " vs " << InferOutputSize() << std::endl;
        ss << "Are you implement InferOutputSize() for " << OpTypeToString(GetOpType()) << "?";
        DLogFatal() << ss.str();
    }

    CheckInput();

    InferOutputMetaInfo();

    if (!SkipDistributedSpecFromPlacementSignature()) {
        InferOutputDistributedSpecFromPlacementSignature();
    }
}

//------------------------------------------------- Topology -----------------------------------------------------------

OperandArray Operator::CreateOutputOperands() {
    DDebugAssert(mOutputOperands.size() == 0);

    size_t expectOutputSize = InferOutputSize();
    for (size_t i = 0; i < expectOutputSize; i++) {
        mOutputOperands.push_back(std::make_shared<Operand>());
    }
    return mOutputOperands;
}

void Operator::UpdateIOOperandTopology() {
    for (const auto& operand : this->GetInputOperands()) {
        operand->AddConsumesOp(this);
    }

    for (const auto& operand : this->GetOutputOperands()) {
        DDebugAssert(operand->GetProducerOp() == nullptr);
        operand->SetProducerOp(this);
    }
}

//----------------------------------------------- Distributed ----------------------------------------------------------

bool Operator::IsDistributedOperator() const {
    DDebugAssert(GetInputSize() + GetOutputSize() > 0);

    if (GetInputSize() > 0) {
        return OperandX()->IsDistributed();
    }

    if (GetOutputSize() > 0) {
        return OperandY()->IsDistributed();
    }

    return false;
}

bool Operator::SkipDistributedSpecFromPlacementSignature() const {
    DAlwaysAssertMsg(GetInputSize() > 0, "Not support when input size > 0, override this funciton when input size = 0");

    return !GetInputOperand(0)->IsDistributed();
}

void Operator::InferOutputDistributedSpecFromPlacementSignature() const {
    DAlwaysAssertMsg(GetInputSize() > 0, "Not support when input size > 0, override this funciton when input size = 0");
    DeviceMesh outputDeviceMesh(GetInputOperand(0)->GetDeviceMesh());
    DDebugAssert(outputDeviceMesh.Count() > 1);
    PlacementSignature placementSignature = GetPlacementSignature();

    std::vector<PlacementSeq> inputPlacementSeq;
    for (size_t i = 0; i < GetInputSize(); i++) {
        inputPlacementSeq.push_back(GetInputOperand(i)->GetPlacementSeq());
    }
    std::vector<PlacementSeq> outputPlacementSeq;
    // TODO: temp code
    bool keepSubSplitCoordinates = true;
    if (GetOpType() == OperatorType::kGetItem) {
        keepSubSplitCoordinates = DerivedCast<GetItemOp, Operator>(this)->GetKeepSubSplitCoordinates();
    }
    if (!placementSignature.Match(inputPlacementSeq, outputPlacementSeq, keepSubSplitCoordinates)) {
        std::stringstream ss;
        ss << "Match placement signature for " << OpTypeToString(GetOpType()) << " operator failed." << std::endl;
        ss << "Placements of input tensor: " << std::endl;
        for (size_t i = 0; i < inputPlacementSeq.size(); i++) {
            ss << "    in" << i << ": " << inputPlacementSeq[i] << std::endl;
        }
        ss << "Placement signature is: " << std::endl;
        ss << placementSignature.ToString();
        throw std::invalid_argument(ss.str());
    }

    size_t outputSize = GetOutputSize();
    for (size_t i = 0; i < outputSize; i++) {
        DeviceMesh deviceMesh = outputDeviceMesh;
        const PlacementSeq& placementSeq = outputPlacementSeq[i];

        Operand* operand = GetOutputOperand(i);
        Shape shape = operand->GetShape();
        if (!DistributedSpec::CheckShapeValid(shape, deviceMesh, placementSeq)) {
            throw std::invalid_argument("Operator output shape not compatibale with distribute spec");
        }
        operand->SetDeviceMeshAndPlacementSeq(deviceMesh, placementSeq);
    }
}

PlacementSignature Operator::GetPlacementSignature() const {
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    return builder.Finish();
}

OperatorCost Operator::GetOperatorCost() const {
    // Memory-bound default: delegate to OperatorCost, which sums the bytes of every non-null input and
    // output operand. Compute-bound operators override GetOperatorCost() and return OperatorCost::Compute(flops).
    return OperatorCost::FromOperands(GetInputOperands(), GetOutputOperands());
}

// From input or output operand
const std::unordered_set<int64_t>& Operator::GetDistributedDeviceIdSet() const {
    DAlwaysAssert(IsDistributedOperator());

    if (GetInputSize() > 0) {
        return OperandX()->GetDistributedDeviceIdSet();
    } else {
        DAlwaysAssert(GetOutputSize() > 0);
        return OperandY()->GetDistributedDeviceIdSet();
    }
}

size_t Operator::GetValidDim(const Shape& shape, int64_t dim) {
    size_t numAixs = shape.NumAxis();
    int64_t result = dim < 0 ? dim + numAixs : dim;
    if (result < 0 || result > static_cast<int64_t>(numAixs)) {
        std::stringstream ss;
        ss << "Dimension out of range (expected to be in range of [-" << numAixs << ", " << numAixs - 1 << "], but got "
           << dim << ")";
        throw std::invalid_argument(ss.str());
    }
    return result;
}

//---------------------------------------------- Kernel Compute --------------------------------------------------------

void Operator::InferOperatorAssignInfo() {
    DDebugAssert(mOperatorAssignInfo.NumKernelForThisOp() == 0);

    // Input
    for (size_t i = 0; i < GetInputSize(); i++) {
        const auto& deviceMesh = GetInputOperand(i)->GetDeviceMesh();
        DDebugAssert(deviceMesh.NumAxis() > 0);
        DeviceKind deviceKind = deviceMesh.GetDeviceKind();
        const auto& deviceIdSet = deviceMesh.GetDeviceIdSet();

        for (auto it : deviceIdSet) {
            KernelStreamKey streamKey;
            streamKey.Init(deviceKind, it, KernelStreamType::kCompute);
            mOperatorAssignInfo.Insert(streamKey);
        }
    }

    // Output
    if (GetOutputSize() > 0) {
        const auto& deviceMesh = GetOutputOperand(0)->GetDeviceMesh();
        DDebugAssert(deviceMesh.NumAxis() > 0);
        for (size_t i = 1; i < GetOutputSize(); i++) {
            DAlwaysAssert(deviceMesh == GetOutputOperand(i)->GetDeviceMesh());
        }

        DeviceKind deviceKind = deviceMesh.GetDeviceKind();
        const auto& deviceIdSet = deviceMesh.GetDeviceIdSet();
        for (auto it : deviceIdSet) {
            KernelStreamKey streamKey;
            streamKey.Init(deviceKind, it, KernelStreamType::kCompute);
            mOperatorAssignInfo.Insert(streamKey);
        }
    }

    DDebugAssert(mOperatorAssignInfo.NumKernelForThisOp() > 0);
}

void Operator::Compute(const TorchTensorOptArray&, TorchTensorArray&) const {
    DLogFatal() << "Override this funciton in derived class for operator: " << OpTypeToString(GetOpType());
    DUnimplemented();
}

OperatorSerializationPack Operator::GetOperatorSerializationPack() {
    OperatorSerializationPack pack;
    pack.opName = mOpName;
    pack.uniqueId = GetUniqueId();
    pack.opParam = mOpParam;
    for (auto operand : mInputOperands) {
        pack.uintInputOperands.push_back(reinterpret_cast<uintptr_t>(operand.get()));
    }
    for (auto operand : mOutputOperands) {
        pack.uintOutputOperands.push_back(reinterpret_cast<uintptr_t>(operand.get()));
    }
    return pack;
}

}  // namespace core
}  // namespace dtorch
