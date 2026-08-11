/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>

#include "dtorch/common/debug.h"  // DTORCH_FORCEINLINE
#include "dtorch/core/operand.h"  // Operand, OperandArray (pulls in Shape / DataKind / DataKindSize)

namespace dtorch {
namespace core {

// Cost estimation for an Operator given its current input/output shapes.
//
// Usually only one channel carries a value: compute-bound ops (matmul, conv, sdpa, linear, outer,
// ...) report FLOPs via Compute(), while memory-bound ops (add, copy, reshape, activation, ...)
// report memory traffic in bytes via Memory(). The other channel stays std::nullopt, and HasCost()
// is false when neither is set. Used downstream for compute/bandwidth utilization (roofline) analysis.
//
// An operator that is neither purely compute- nor purely memory-bound (e.g. a fused kernel that wants
// both numbers for a full roofline view) may set BOTH channels via ComputeAndMemory(); ToString() then
// renders the two together.
//
// OperatorCost owns the cost-construction logic so operators stay thin: a subclass override only has to
// pick a factory (Compute / Memory) and feed it a number, and the base Operator::GetOperatorCost()
// delegates the generic memory-bound default to FromOperands().
class OperatorCost {
public:
    //--------------------------------------------- Factories --------------------------------------------------------

    // Compute-bound cost: report FLOPs of arithmetic work. matmul / conv / sdpa / linear / outer ...
    static OperatorCost Compute(int64_t flops);

    // Memory-bound cost: report bytes of memory traffic. add / copy / reshape / activation ...
    static OperatorCost Memory(int64_t bandwidthBytes);

    // Mixed cost: an operator that is neither purely compute- nor purely memory-bound reports BOTH its
    // FLOPs and its memory traffic (e.g. a fused kernel needing both numbers for a full roofline view).
    static OperatorCost ComputeAndMemory(int64_t flops, int64_t bandwidthBytes);

    // Memory-bound default for a generic Operator: total bytes = sum of every non-null input and
    // output operand (element count * element size). Null-shape operands are skipped so that absent
    // optional inputs (e.g. a null bias) contribute 0 bytes. This is the base Operator::GetOperatorCost().
    static OperatorCost FromOperands(const OperandArray& inputs, const OperandArray& outputs);

public:
    OperatorCost() : mFlops(std::nullopt), mBandwidthBytes(std::nullopt) {}

    DTORCH_FORCEINLINE bool IsComputeBound() const { return mFlops.has_value(); }
    DTORCH_FORCEINLINE bool IsMemoryBound() const { return mBandwidthBytes.has_value(); }
    DTORCH_FORCEINLINE bool HasCost() const { return mFlops.has_value() || mBandwidthBytes.has_value(); }

    DTORCH_FORCEINLINE const std::optional<int64_t>& Flops() const { return mFlops; }
    DTORCH_FORCEINLINE const std::optional<int64_t>& BandwidthBytes() const { return mBandwidthBytes; }

    std::string ToString() const;
    friend std::ostream& operator<<(std::ostream& os, const OperatorCost& cost);

private:
    std::optional<int64_t> mFlops;           // compute work (FLOPs); nullopt unless compute work is reported
    std::optional<int64_t> mBandwidthBytes;  // memory traffic (bytes); nullopt unless memory traffic is reported
};

}  // namespace core
}  // namespace dtorch
