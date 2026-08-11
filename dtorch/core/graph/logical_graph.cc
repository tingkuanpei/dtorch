/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "logical_graph.h"

namespace dtorch {
namespace core {

LogicalGraph::LogicalGraph() : mName(""), mOperatorMap(), mOperandMap() {}

LogicalGraph::~LogicalGraph() {
    // Operand will use mOperandMap in destructor, we have to manual destruct operand before destruct map

    while (!mOperandMap.empty()) {
        mOperandMap.erase(mOperandMap.begin());
    }

    while (!mOperatorMap.empty()) {
        mOperatorMap.erase(mOperatorMap.begin());
    }
}

void LogicalGraph::AddOperator(std::unique_ptr<Operator> op) {
    Operator* opPtr = op.get();
    DDebugAssert(mOperatorMap.find(opPtr) == mOperatorMap.end());

    mOperatorMap[opPtr] = std::move(op);
}

void LogicalGraph::AddOperand(const std::shared_ptr<Operand>& operand) {
    DDebugAssert(mOperandMap.find(operand.get()) == mOperandMap.end());
    mOperandMap[operand.get()] = std::move(operand);
}

void LogicalGraph::DeleteOperator(const Operator* op) {
    const auto& it = mOperatorMap.find(op);
    DDebugAssert(it != mOperatorMap.end());
    mOperatorMap.erase(it);
}

void LogicalGraph::DeleteOperand(const Operand* operand) {
    const auto& it = mOperandMap.find(operand);
    DDebugAssert(it != mOperandMap.end());
    mOperandMap.erase(it);
}

}  // namespace core
}  // namespace dtorch
