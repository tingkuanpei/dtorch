/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "./operator.h"
#include "dtorch/core/graph/logical_graph.h"

namespace dtorch {
namespace core {

using SubGraphParam = NoElementOpParam<OperatorType::kSubGraph>;

class SubGraphOp : public Operator {
public:
public:
    SubGraphOp(std::shared_ptr<OpParam> opParam) : Operator(opParam), mLogicalGraph() {}

    std::vector<const Operator*> DFSSequence() { return std::vector<const Operator*>(); }

private:
    LogicalGraph mLogicalGraph;
};

}  // namespace core
}  // namespace dtorch
