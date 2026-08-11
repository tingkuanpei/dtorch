/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <unordered_set>

#include "dtorch/common/debug.h"
#include "dtorch/core/global_id_manager.h"
#include "dtorch/core/operand.h"
#include "dtorch/core/operators/operator_assign_info.h"
#include "dtorch/core/operators/operator_cost.h"
#include "dtorch/core/operators/placement_signature.h"
#include "dtorch/external/torch/torch_util.h"
#include "operator_param.h"

namespace dtorch {
namespace core {

using external::torch::TorchTensorArray;
using external::torch::TorchTensorOptArray;

class OperatorSerializationPack;

class Operator {
public:
    Operator(std::shared_ptr<OpParam> opParamPtr);

    virtual ~Operator() = default;

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(Operator);

    DTORCH_FORCEINLINE void SetOpName(const std::string& name) { mOpName = name; }

    DTORCH_FORCEINLINE const std::string& GetOpName() const noexcept { return mOpName; }

    DTORCH_FORCEINLINE void AssignUniqueId() { mUniqueId = OperatorIdManager::GetSingleton().GetUniqueId(); }

    DTORCH_FORCEINLINE void SetUniqueId(uint64_t uniqueId) {
        DAlwaysAssert(uniqueId != OperatorIdManager::kNoValue);
        mUniqueId = uniqueId;
    }

    DTORCH_FORCEINLINE uint64_t GetUniqueId() const noexcept {
        DAlwaysAssert(mUniqueId != OperatorIdManager::kNoValue);
        return mUniqueId;
    }

    DTORCH_FORCEINLINE const OperatorType& GetOpType() const noexcept { return mOpType; }

    template <typename DerivedOpParam>
    DTORCH_FORCEINLINE const DerivedOpParam& GetOpParam() const {
        return *(DerivedCast<DerivedOpParam, OpParam>(GetOpParam()));
    }

    template <typename DerivedOpParam>
    DTORCH_FORCEINLINE DerivedOpParam& GetOpParam() {
        return *(DerivedCast<DerivedOpParam, OpParam>(GetOpParam()));
    }

    DTORCH_FORCEINLINE const OpParam* GetOpParam() const { return mOpParam.get(); }

    DTORCH_FORCEINLINE OpParam* GetOpParam() { return const_cast<OpParam*>(mOpParam.get()); }

    DTORCH_FORCEINLINE virtual std::string GetDescribeString() const { return OpTypeToString(GetOpType()); }

    //-------------------------------------------- Check Input ---------------------------------------------------------

    virtual void CheckInput() const;

    virtual bool IsRequireInputSameDataKind() const { return true; }

    void CheckInputSameDataKind() const;

    void CheckInputAllDistributedOrNot() const;

    void CheckInputSameDeviceMesh() const;

    void CheckInputDistributedSpec() const;

    //----------------------------------------------- Infer ------------------------------------------------------------

    void Infer();

    // Create and infer output operand(shape, stride, data kind and so on, except device and distribute operand info).
    void InferOutput();

    // Check input and infer output shape
    // 1. Check input operand information: shape, stride, data kind and so on.
    // 2. Infer output operand info: shape, stride, data kind and so on, except device and distribute operand info,
    // override this function for sub class.
    virtual void InferOutputMetaInfo() const = 0;

    // Infer output size
    // 1. use mOpParamPtr to infer output size
    // 2. if output size depend on input data, set max possible output size.
    // If output size not equal to 1, override this function for sub class.
    virtual size_t InferOutputSize() const { return 1; };

    //--------------------------------------------- Topology -----------------------------------------------------------

    DTORCH_FORCEINLINE void SetInputOperands(const OperandArray& inputOperand) { mInputOperands = inputOperand; }

    // Create output operand according to InferOutputSize()
    OperandArray CreateOutputOperands();

    virtual void UpdateIOOperandTopology();

    DTORCH_FORCEINLINE Operand* GetInputOperand(size_t index) const noexcept {
        DDebugAssert(index < mInputOperands.size());
        return mInputOperands[index].get();
    }

    DTORCH_FORCEINLINE Operand* GetOutputOperand(size_t index) const noexcept {
        DDebugAssert(index < mOutputOperands.size());
        return mOutputOperands[index].get();
    }

    DTORCH_FORCEINLINE Operand* OperandX() const { return GetInputOperand(0); }

    DTORCH_FORCEINLINE Operand* OperandWeight() const { return GetInputOperand(1); }

    DTORCH_FORCEINLINE Operand* OperandBias() const { return GetInputOperand(2); }

    DTORCH_FORCEINLINE Operand* OperandA() const { return GetInputOperand(0); }

    DTORCH_FORCEINLINE Operand* OperandB() const { return GetInputOperand(1); }

    DTORCH_FORCEINLINE Operand* OperandC() const { return GetInputOperand(2); }

    DTORCH_FORCEINLINE Operand* OperandD() const { return GetInputOperand(3); }

    DTORCH_FORCEINLINE Operand* OperandE() const { return GetInputOperand(4); }

    DTORCH_FORCEINLINE Operand* OperandY() const { return GetOutputOperand(0); }

    DTORCH_FORCEINLINE const OperandArray& GetInputOperands() const noexcept { return mInputOperands; }

    DTORCH_FORCEINLINE const OperandArray& GetOutputOperands() const noexcept { return mOutputOperands; }

    DTORCH_FORCEINLINE size_t GetInputSize() const noexcept { return mInputOperands.size(); }

    DTORCH_FORCEINLINE size_t GetOutputSize() const noexcept { return mOutputOperands.size(); }

    //------------------------------------------- Distributed ----------------------------------------------------------

    bool IsDistributedOperator() const;

    virtual bool SkipDistributedSpecFromPlacementSignature() const;

    virtual void InferOutputDistributedSpecFromPlacementSignature() const;

    virtual PlacementSignature GetPlacementSignature() const;

    // From input or output operand
    const std::unordered_set<int64_t>& GetDistributedDeviceIdSet() const;

    //------------------------------------------ Kernel Compute --------------------------------------------------------

    virtual void InferOperatorAssignInfo();

    DTORCH_FORCEINLINE const OperatorAssignInfo& GetOperatorAssignInfo() const noexcept { return mOperatorAssignInfo; }

    DTORCH_FORCEINLINE OperatorAssignInfo& GetOperatorAssignInfo() noexcept { return mOperatorAssignInfo; }

    DTORCH_FORCEINLINE void SetOperatorAssignInfo(const OperatorAssignInfo& opAssignInfo) noexcept {
        mOperatorAssignInfo = opAssignInfo;
    }

    virtual void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const;

    OperatorSerializationPack GetOperatorSerializationPack();

    //--------------------------------------------- Cost Estimation ---------------------------------------------------

    // Compute cost (FLOPs for compute-bound ops, bandwidthBytes for memory-bound ops) given current shapes.
    // Precondition: called after Infer(), so every input AND output operand shape is already inferred
    // and ready to read — overrides should read operand shapes directly rather than recompute them.
    // Base default is memory-bound: bandwidthBytes = sum of all non-null input + output operand bytes.
    // Override in compute-bound operators to report `flops` instead.
    virtual OperatorCost GetOperatorCost() const;

public:
    static size_t GetValidDim(const Shape& shape, int64_t dim);

protected:
    std::string mOpName;
    uint64_t mUniqueId;
    OperatorType mOpType;
    std::shared_ptr<OpParam> mOpParam;

    // Topology
    OperandArray mInputOperands;
    OperandArray mOutputOperands;

    // How to create kernel
    OperatorAssignInfo mOperatorAssignInfo;
};

}  // namespace core
}  // namespace dtorch
