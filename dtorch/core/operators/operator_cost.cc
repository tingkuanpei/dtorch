/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "operator_cost.h"

#include <sstream>

namespace dtorch {
namespace core {

OperatorCost OperatorCost::Compute(int64_t flops) {
    OperatorCost cost;
    cost.mFlops = flops;
    return cost;
}

OperatorCost OperatorCost::Memory(int64_t bandwidthBytes) {
    OperatorCost cost;
    cost.mBandwidthBytes = bandwidthBytes;
    return cost;
}

OperatorCost OperatorCost::ComputeAndMemory(int64_t flops, int64_t bandwidthBytes) {
    OperatorCost cost;
    cost.mFlops = flops;
    cost.mBandwidthBytes = bandwidthBytes;
    return cost;
}

OperatorCost OperatorCost::FromOperands(const OperandArray& inputs, const OperandArray& outputs) {
    // Memory-bound default: total bytes touched = sum over all non-null input and output operands.
    // Null-shape operands are skipped so absent optional inputs (e.g. a null bias) contribute 0 bytes
    // rather than a spurious (-100 * elementSize) term.
    int64_t totalBytes = 0;
    auto accumulate = [&totalBytes](const Operand* operand) {
        if (operand->IsNullTensorShape()) {
            return;
        }
        totalBytes += static_cast<int64_t>(operand->GetShape().Count()) *
                      static_cast<int64_t>(DataKindSize(operand->GetDataKind()));
    };
    for (const auto& operand : inputs) {
        accumulate(operand.get());
    }
    for (const auto& operand : outputs) {
        accumulate(operand.get());
    }
    return Memory(totalBytes);
}

std::string OperatorCost::ToString() const {
    // Render each present channel independently. The two are not mutually exclusive: a mixed op may set
    // both FLOPs and bandwidth, so we must not gate one on the absence of the other.
    std::stringstream ss;
    ss << "OperatorCost[";
    bool first = true;
    auto append = [&](const char* key, int64_t value) {
        if (!first) {
            ss << ", ";
        }
        ss << key << "=" << value;
        first = false;
    };
    if (mFlops.has_value()) {
        append("flops", *mFlops);
    }
    if (mBandwidthBytes.has_value()) {
        append("bandwidthBytes", *mBandwidthBytes);
    }
    if (first) {
        ss << "none";
    }
    ss << "]";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const OperatorCost& cost) { return os << cost.ToString(); }

}  // namespace core
}  // namespace dtorch
