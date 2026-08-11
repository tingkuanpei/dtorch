/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <unordered_map>

#include "dtorch/common/config.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/operand.h"
#include "dtorch/core/operators/operator.h"

namespace dtorch {
namespace core {

class LogicalGraph {
public:
    LogicalGraph();

    ~LogicalGraph();

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(LogicalGraph);

    DTORCH_FORCEINLINE void SetName(const std::string& name) { mName = name; }
    DTORCH_FORCEINLINE const std::string& GetName() const noexcept { return mName; }

    void AddOperator(std::unique_ptr<Operator> op);
    void AddOperand(const std::shared_ptr<Operand>& operand);

    // Only remove Operator and Operand, not update graph topology
    void DeleteOperator(const Operator* op);
    void DeleteOperand(const Operand* operand);

    DTORCH_FORCEINLINE bool CountOperator(const Operator* op) const noexcept { return mOperatorMap.count(op) > 0; }

    DTORCH_FORCEINLINE bool CountOperand(const Operand* operand) const noexcept {
        return mOperandMap.count(operand) > 0;
    }

    DTORCH_FORCEINLINE const std::shared_ptr<Operator>& GetOperator(const Operator* op) const {
        DDebugAssert(mOperatorMap.count(op) > 0);
        return mOperatorMap.at(op);
    }

    DTORCH_FORCEINLINE const std::shared_ptr<Operand>& GetOperand(const Operand* operand) const {
        DDebugAssert(CountOperand(operand));
        return mOperandMap.at(operand);
    }

    DTORCH_FORCEINLINE const auto& GetOperatorMap() const noexcept { return mOperatorMap; }

    DTORCH_FORCEINLINE const auto& GetOperandMap() const noexcept { return mOperandMap; }

protected:
    std::string mName;
    std::unordered_map<const Operator*, std::shared_ptr<Operator>> mOperatorMap;
    std::unordered_map<const Operand*, std::shared_ptr<Operand>> mOperandMap;
};

}  // namespace core
}  // namespace dtorch
