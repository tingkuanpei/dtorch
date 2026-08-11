/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <sstream>

#include "dtorch/api/cpp/data_kind.h"
#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/int_or_int_array.h"
#include "dtorch/api/cpp/shape.h"
#include "dtorch/core/operand.h"
#include "dtorch/core/operators/operator.h"
#include "dtorch/core/operators/operator_cost.h"
#include "dtorch/core/operators/standard/conv_op.h"
#include "dtorch/core/operators/standard/linear_op.h"
#include "dtorch/core/operators/standard/matmul_op.h"
#include "dtorch/core/operators/standard/outer_op.h"
#include "dtorch/core/operators/standard/scaled_dot_product_attention_op.h"
#include "dtorch/core/operators/system/sync_op.h"
#include "test.h"

using namespace dtorch::api::cpp;
using namespace dtorch::core;

namespace {

// Build a local (single-CPU-device) operand with the given shape/kind. Cost estimation only reads
// shape + data kind, so a 1-device mesh with an empty placement sequence is sufficient.
std::shared_ptr<Operand> MakeOperand(const Shape& shape, DataKind kind = DataKind::kFloat32) {
    DeviceMesh mesh(Device(DeviceKind::kCpu, 0));
    return std::make_shared<Operand>(shape, kind, mesh, PlacementSeq());
}

// GetOperatorCost() is contractually called after Infer(): set up inputs, create the output operands and
// run shape inference so every operand shape is ready before the cost is read.
void PrepareForCost(Operator& op, const OperandArray& inputs) {
    op.SetInputOperands(inputs);
    op.CreateOutputOperands();
    op.InferOutputMetaInfo();
}

}  // namespace

// ============================================================
// OperatorCost class unit tests
// ============================================================

TEST(OperatorCostTest, DefaultIsEmpty) {
    OperatorCost cost;
    EXPECT_FALSE(cost.HasCost());
    EXPECT_FALSE(cost.IsComputeBound());
    EXPECT_FALSE(cost.IsMemoryBound());
    EXPECT_FALSE(cost.Flops().has_value());
    EXPECT_FALSE(cost.BandwidthBytes().has_value());
}

TEST(OperatorCostTest, ComputeFactory) {
    OperatorCost cost = OperatorCost::Compute(123456);
    EXPECT_TRUE(cost.IsComputeBound());
    EXPECT_FALSE(cost.IsMemoryBound());
    ASSERT_TRUE(cost.Flops().has_value());
    EXPECT_EQ(*cost.Flops(), 123456);
    EXPECT_FALSE(cost.BandwidthBytes().has_value());
}

TEST(OperatorCostTest, MemoryFactory) {
    OperatorCost cost = OperatorCost::Memory(4096);
    EXPECT_TRUE(cost.IsMemoryBound());
    EXPECT_FALSE(cost.IsComputeBound());
    ASSERT_TRUE(cost.BandwidthBytes().has_value());
    EXPECT_EQ(*cost.BandwidthBytes(), 4096);
    EXPECT_FALSE(cost.Flops().has_value());
}

TEST(OperatorCostTest, ComputeAndMemoryFactory) {
    // A mixed op reports both channels; neither gate excludes the other.
    OperatorCost cost = OperatorCost::ComputeAndMemory(1000, 2000);
    EXPECT_TRUE(cost.IsComputeBound());
    EXPECT_TRUE(cost.IsMemoryBound());
    EXPECT_TRUE(cost.HasCost());
    ASSERT_TRUE(cost.Flops().has_value());
    ASSERT_TRUE(cost.BandwidthBytes().has_value());
    EXPECT_EQ(*cost.Flops(), 1000);
    EXPECT_EQ(*cost.BandwidthBytes(), 2000);
}

TEST(OperatorCostTest, FromOperandsSumsBytes) {
    // Two float32 inputs (100 elems each) + one float32 output (100 elems).
    OperandArray inputs = {MakeOperand(Shape({100})), MakeOperand(Shape({100}))};
    OperandArray outputs = {MakeOperand(Shape({100}))};
    OperatorCost cost = OperatorCost::FromOperands(inputs, outputs);
    EXPECT_TRUE(cost.IsMemoryBound());
    ASSERT_TRUE(cost.BandwidthBytes().has_value());
    EXPECT_EQ(*cost.BandwidthBytes(), 3 * 100 * 4);  // 3 operands * 100 elems * 4 bytes/elem
}

TEST(OperatorCostTest, FromOperandsSkipsNullShape) {
    // A null-shape operand (e.g. an absent bias) must contribute 0 bytes, not (-100 * size).
    OperandArray inputs = {MakeOperand(Shape({64})), MakeOperand(Shape::GetNullShape(), DataKind::kFloat32)};
    OperatorCost cost = OperatorCost::FromOperands(inputs, OperandArray{});
    ASSERT_TRUE(cost.BandwidthBytes().has_value());
    EXPECT_EQ(*cost.BandwidthBytes(), 64 * 4);  // only the first operand counts
}

TEST(OperatorCostTest, FromOperandsRespectsDataKind) {
    // float64 input: 8 bytes per element.
    OperandArray inputs = {MakeOperand(Shape({50}), DataKind::kFloat64)};
    OperatorCost cost = OperatorCost::FromOperands(inputs, OperandArray{});
    ASSERT_TRUE(cost.BandwidthBytes().has_value());
    EXPECT_EQ(*cost.BandwidthBytes(), 50 * 8);
}

TEST(OperatorCostTest, ToStringAndStream) {
    EXPECT_EQ(OperatorCost::Compute(7).ToString(), "OperatorCost[flops=7]");
    EXPECT_EQ(OperatorCost::Memory(9).ToString(), "OperatorCost[bandwidthBytes=9]");
    // A mixed op renders both channels together — neither may be silently dropped.
    EXPECT_EQ(OperatorCost::ComputeAndMemory(7, 9).ToString(), "OperatorCost[flops=7, bandwidthBytes=9]");
    EXPECT_EQ(OperatorCost().ToString(), "OperatorCost[none]");

    std::stringstream ss;
    ss << OperatorCost::Compute(42);
    EXPECT_EQ(ss.str(), "OperatorCost[flops=42]");
}

// ============================================================
// Compute-bound operator overrides (report FLOPs)
// ============================================================

TEST(OperatorCostOverrideTest, MatmulFlops) {
    // A: [M=128, K=256], B: [K=256, N=512] → 2*M*N*K MACs.
    MatmulOp op(std::make_shared<MatmulParam>());
    PrepareForCost(op, {MakeOperand(Shape({128, 256})), MakeOperand(Shape({256, 512}))});
    OperatorCost cost = op.GetOperatorCost();
    EXPECT_TRUE(cost.IsComputeBound());
    ASSERT_TRUE(cost.Flops().has_value());
    EXPECT_EQ(*cost.Flops(), 2 * 128 * 256 * 512);
}

TEST(OperatorCostOverrideTest, MatmulBatchedFlops) {
    // bmm: A: [B=4, M=8, K=16], B: [4, 16, 32] → batch=4.
    MatmulOp op(std::make_shared<MatmulParam>());
    PrepareForCost(op, {MakeOperand(Shape({4, 8, 16})), MakeOperand(Shape({4, 16, 32}))});
    OperatorCost cost = op.GetOperatorCost();
    ASSERT_TRUE(cost.Flops().has_value());
    EXPECT_EQ(*cost.Flops(), 2 * 4 * 8 * 32 * 16);
}

TEST(OperatorCostOverrideTest, LinearFlops) {
    // X: [B=10, inFeatures=64], W: [outFeatures=96, 64], null bias → 2*B*in*out.
    LinearOp op(std::make_shared<LinearParam>());
    PrepareForCost(op,
                   {MakeOperand(Shape({10, 64})), MakeOperand(Shape({96, 64})), MakeOperand(Shape::GetNullShape())});
    OperatorCost cost = op.GetOperatorCost();
    EXPECT_TRUE(cost.IsComputeBound());
    ASSERT_TRUE(cost.Flops().has_value());
    EXPECT_EQ(*cost.Flops(), 2 * 10 * 64 * 96);
}

TEST(OperatorCostOverrideTest, OuterFlops) {
    // a: [7], b: [11] → 7*11 (1 FLOP/element, no 2x MAC factor).
    OuterOp op(std::make_shared<OuterParam>());
    PrepareForCost(op, {MakeOperand(Shape({7})), MakeOperand(Shape({11}))});
    OperatorCost cost = op.GetOperatorCost();
    ASSERT_TRUE(cost.Flops().has_value());
    EXPECT_EQ(*cost.Flops(), 7 * 11);
}

TEST(OperatorCostOverrideTest, SdpaFlops) {
    // Q: [batch=2, heads=8, seqQ=16, headDim=64], K: [2, 8, seqKv=20, 64], V: [2, 8, 20, E_v=64].
    // → 2 * batch * heads * seqQ * seqKv * (headDim + E_v).
    SdpaOp op(std::make_shared<SdpaParam>());
    PrepareForCost(op, {MakeOperand(Shape({2, 8, 16, 64})),    // Q
                        MakeOperand(Shape({2, 8, 20, 64})),    // K
                        MakeOperand(Shape({2, 8, 20, 64})),    // V
                        MakeOperand(Shape::GetNullShape())});  // optional attn_mask (unused by cost)
    OperatorCost cost = op.GetOperatorCost();
    EXPECT_TRUE(cost.IsComputeBound());
    ASSERT_TRUE(cost.Flops().has_value());
    EXPECT_EQ(*cost.Flops(), 2 * 2 * 8 * 16 * 20 * (64 + 64));
}

TEST(OperatorCostOverrideTest, ConvFlops) {
    // NCHW input [1, 3, 8, 8], weight [outC=4, inC/group=3, 3, 3], kValid 3x3 stride 1 dilation 1.
    // Output spatial = 6x6 (kValid: (8 - 3)/1 + 1); inCPerGroup = 3; → 2*1*4*6*6*3*3*3.
    auto param = std::make_shared<ConvParam>(IntOrIntArray({1, 1}), 1, IntOrIntArray({3, 3}), PaddingType::kValid,
                                             IntOrIntArray({0}), IntOrIntArray({1, 1}), OperatorFormat::kNCHW);
    ConvOp op(param);
    PrepareForCost(op, {MakeOperand(Shape({1, 3, 8, 8})),      // input
                        MakeOperand(Shape({4, 3, 3, 3})),      // weight
                        MakeOperand(Shape::GetNullShape())});  // null bias
    OperatorCost cost = op.GetOperatorCost();
    EXPECT_TRUE(cost.IsComputeBound());
    ASSERT_TRUE(cost.Flops().has_value());
    EXPECT_EQ(*cost.Flops(), 2 * 1 * 4 * 6 * 6 * 3 * 3 * 3);
}

// ============================================================
// Memory-bound default + system ops
// ============================================================

TEST(OperatorCostOverrideTest, SyncOpHasNoCost) {
    // SyncOp is pure control flow: its override returns an empty OperatorCost (no compute, no data moved).
    SyncOp op(std::make_shared<SyncParam>());
    OperatorCost cost = op.GetOperatorCost();
    EXPECT_FALSE(cost.HasCost());
}
