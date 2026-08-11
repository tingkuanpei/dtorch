/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "matmul_op.h"

#include "dtorch/common/debug.h"
#include "dtorch/core/operators/standard/broadcast_binary_op.h"

// MatmulOp — Matrix Multiplication
//
// == 功能描述 (Functionality) ==
// 实现 torch.matmul / torch.mm / torch.bmm 的语义，支持多种输入维度组合
// 和批量广播。
//
// 支持的维度组合：
//   - 2D × 2D:  标准矩阵乘法 [M, K] × [K, N] → [M, N]
//   - 1D × 2D:  向量乘矩阵      [K]    × [K, N] → [N]
//   - 2D × 1D:  矩阵乘向量      [M, K] × [K]    → [M]
//   - 1D × 1D:  点积            [K]    × [K]    → scalar
//   - ND × ND:  批量矩阵乘法，batch 维度支持 broadcast (N >= 3)
//   - ND × 2D / 2D × ND / ND × 1D / 1D × ND: 批量+broadcast 混合
//
// 数学定义：
//   2D × 2D 情况：Y[m,n] = sum_k A[m,k] * B[k,n]
//   ND 情况：batch 维度独立 broadcast 后，对最后两维执行矩阵乘法。
//   1D 输入在计算前自动视为 1×K (1D×2D) 或 K×1 (2D×1D)。
//
// 边界条件：
//   - 输入必须为 float / complex 类型（不支持 integer matmul）
//   - 1D×1D 点积要求两个向量长度相等，输出为标量
//   - ND×MD 的 batch 维度必须 broadcast 兼容
//
// == 输出 Shape (Output Shape) ==
// 输出 Shape 分为四步推导：
//
// Step 1 — 拆分 broadcast 与 compute 维度：
//   将每个输入 shape 拆分为 (broadcast_dims, compute_dims)：
//     - compute_dims = shape 的最后 2 维（不足 2 维时取实际维数）
//     - broadcast_dims = shape 去掉 compute_dims 后的部分
//   例如：[2, 3, 8, 16] → broadcast=[2,3], compute=[8,16]
//         [16]          → broadcast=[],     compute=[16]
//
// Step 2 — Batch 维度广播：
//   对 broadcast_dims 执行 BroadcastOutputShape：
//     broadcastShapeY = BroadcastOutputShape(broadcastShapeA, broadcastShapeB)
//
// Step 3 — Compute 维度矩阵乘法：
//   根据 compute 维度的秩组合推导 compute 输出 shape：
//     - [M, K] × [K, N] → [M, N]   (2D × 2D)
//     - [K]    × [K, N] → [N]       (1D × 2D)
//     - [M, K] × [K]    → [M]       (2D × 1D)
//     - [K]    × [K]    → scalar    (1D × 1D)
//
// Step 4 — 合并：
//   outputShape = MergeShape(broadcastShapeY, computeShapeY)
//
// == Placements 推导 (PlacementSignature) ==
// PlacementSignature 分两部分处理：batch 维度复用 BroadcastBinaryOp 的规则，
// compute 维度按维度组合分别定义映射规则。
//
// 记号约定：
//   设输入 A 的完整 shape 为 [B_A..., M, K]，输入 B 的完整 shape 为 [B_B..., K, N]：
//     mIdx  = inputAShape.NumAxis() - 2   (A 的行维度)
//     kIdxA = inputAShape.NumAxis() - 1   (A 的规约维度)
//     kIdxB = inputBShape.NumAxis() - 2   (B 的规约维度)
//     nIdx  = inputBShape.NumAxis() - 1   (B 的列维度)
//     对应输出 Y 的 shape 为 [B_Y..., M, N]（B_Y 为广播结果）。
//
// Batch 维度规则：
//   BroadcastBinaryOp::AddBroadcastBinaryOpPlacementSignature 处理 batch 维度的
//   Shard 传播：独有维度从一侧传播，共享维度两侧 Shard 对齐传播。
//
// Compute 维度 PlacementSignature 参考表：
//
// ── 2D × 2D (GEMM): [M, K] × [K, N] → [M, N] ──
// ---------------------
// |  A  |  B  |  Y  |
// ---------------------
// | S(M)|  R  | S(M)|  ← 行维度 Shard 从 A 传播到 Y
// |  R  | S(N)| S(N)|  ← 列维度 Shard 从 B 传播到 Y
// | S(K)| S(K)|  P  |  ← 规约维度两侧 Shard → Partial (需 all-reduce)
// | S(M)| S(K)|  P  |  ← 边缘情况：M Shard + K_B Shard → Partial
// | S(K)| S(N)|  P  |  ← 边缘情况：K_A Shard + N Shard → Partial
// |  P  |  R  |  P  |  ← Partial 传播
// |  R  |  P  |  P  |  ← Partial 传播
//
// ── 1D × 2D (向量×矩阵): [K] × [K, N] → [N] ──
//   - 匹配维度 = min(rankA, rankB) = 1，仅在 dim 0 上匹配
//   - 输出 N 维度 = outputDimSize - 1 (即输出的最后一维)
//   - Shard(K) 产生 Partial (每个设备持有点积的部分结果)
//
// ── 2D × 1D (矩阵×向量): [M, K] × [K] → [M] ──
//   - 匹配 dim 0: A 的 M 维 + B 的 K 维 → 输出 M 维 (从 A 传播 Shard(M))
//   - 匹配 dim 1: A 的 K 维 + B 的 K 维 → 规约维度 (输出无对应维度)
//   - Shard(K) 产生 Partial
//
// ── 1D × 1D (点积): [K] × [K] → scalar ──
//   - 标量输出无维度，不支持 Shard/Partial
//   - 依赖 Finish() 的 Replicate→Replicate 回退，仅支持全复制

namespace dtorch {
namespace core {

Shape MatmulOp::ComputeShapeForLessTwoDim(const Shape& shapeA, const Shape& shapeB) const {
    DDebugAssert(shapeA.NumAxis() <= 2);
    DDebugAssert(shapeB.NumAxis() <= 2);

    if (shapeA.NumAxis() == 1 && shapeB.NumAxis() == 1) {
        if (shapeA[0] != shapeB[0]) {
            std::stringstream ss;
            ss << "inconsistent tensor size, expected tensor [" << shapeA[0] << "] and src [" << shapeB[0]
               << "] to have the same number of elements, but got " << shapeA[0] << " and " << shapeB[0]
               << " elements respectively";
            throw std::invalid_argument(ss.str());
        }
        return Shape::GetScalarShape();
    } else if (shapeA.NumAxis() == 2 && shapeB.NumAxis() == 2) {
        if (shapeA[1] != shapeB[0]) {
            std::stringstream ss;
            ss << "mat1 and mat2 shapes cannot be multiplied " << shapeA << " and " << shapeB;
            throw std::invalid_argument(ss.str());
        }
        int64_t M = shapeA[0];
        int64_t N = shapeB[1];
        return Shape({M, N});
    } else if (shapeA.NumAxis() == 1 && shapeB.NumAxis() == 2) {
        if (shapeA[0] != shapeB[0]) {
            std::stringstream ss;
            ss << "mat1 and mat2 shapes cannot be multiplied " << shapeA << " and " << shapeB;
            throw std::invalid_argument(ss.str());
        }
        return Shape({shapeB[1]});
    } else if (shapeA.NumAxis() == 2 && shapeB.NumAxis() == 1) {
        if (shapeA[1] != shapeB[0]) {
            std::stringstream ss;
            ss << "size mismatch, got input (" << shapeA[0] << "), mat " << shapeA << ", vec " << shapeB;
            throw std::invalid_argument(ss.str());
        }
        return Shape({shapeA[0]});
    } else {
        std::stringstream ss;
        ss << "MatmulOp: unsupported compute shape combination: " << shapeA << " × " << shapeB;
        throw std::invalid_argument(ss.str());
    }
}

std::tuple<Shape, Shape> MatmulOp::SplitBroadcastShape(const Shape& shape) const {
    Shape resultA, resultB;
    for (size_t i = 0; i < shape.NumAxis(); i++) {
        if (static_cast<int64_t>(i) < static_cast<int64_t>(shape.NumAxis()) - 2) {
            resultA.PushBack(shape[i]);
        } else {
            resultB.PushBack(shape[i]);
        }
    }
    return {resultA, resultB};
}

Shape MatmulOp::MergeShape(const Shape& shapeA, const Shape& shapeB) const {
    Shape result = shapeA;
    for (size_t i = 0; i < shapeB.NumAxis(); i++) {
        result.PushBack(shapeB[i]);
    }
    return result;
}

void MatmulOp::CheckInput() const {
    Operator::CheckInput();

    DDebugAssert(GetInputSize() == 2);

    Operand* inputA = OperandA();
    Operand* inputB = OperandB();

    if (inputA->IsNullTensorShape() || inputB->IsNullTensorShape()) {
        throw std::invalid_argument("MatmulOp: both inputs must be non-null tensors");
    }

    const Shape& shapeA = inputA->GetShape();
    const Shape& shapeB = inputB->GetShape();

    // Split shapes into broadcast and compute parts.
    // Contracted-dim and broadcast validation is centralized here so that
    // InferOutputMetaInfo can skip redundant checks.
    auto [broadcastShapeA, computeShapeA] = SplitBroadcastShape(shapeA);
    auto [broadcastShapeB, computeShapeB] = SplitBroadcastShape(shapeB);

    // Validate broadcast dimension compatibility.
    if (!broadcastShapeA.CanBroadcastWith(broadcastShapeB)) {
        std::stringstream ss;
        ss << "MatmulOp: broadcast dimensions not compatible: " << broadcastShapeA << " vs " << broadcastShapeB;
        throw std::invalid_argument(ss.str());
    }

    // Validate contracted dimension compatibility.
    // computeShapeA's last dim is the contracted dim (K).
    // computeShapeB's first dim is the contracted dim (K) — regardless of whether
    // B is 1D [K] or 2D [K, N].
    int64_t kSizeA = computeShapeA[computeShapeA.NumAxis() - 1];
    int64_t kSizeB = computeShapeB[0];

    if (kSizeA != kSizeB) {
        std::stringstream ss;
        ss << "MatmulOp: contracted dimensions do not match: "
           << "input A shape " << shapeA << " vs input B shape " << shapeB;
        throw std::invalid_argument(ss.str());
    }
}

std::string MatmulOp::GetDescribeString() const {
    std::stringstream ss;
    ss << "Matmul(";
    if (GetInputSize() >= 2) {
        ss << OperandA()->GetShape() << " × " << OperandB()->GetShape();
        if (GetOutputSize() >= 1 && !OperandY()->IsNullTensorShape()) {
            ss << " → " << OperandY()->GetShape();
        }
    }
    ss << ")";
    return ss.str();
}

void MatmulOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 2);

    Operand* inputA = OperandA();
    Operand* inputB = OperandB();
    const Shape& inputAShape = inputA->GetShape();
    const Shape& inputBShape = inputB->GetShape();

    // CheckInput() already validated broadcast and contracted-dim compatibility,
    // so we can proceed directly to shape computation.
    auto [broadcastShapeA, computeShapeA] = SplitBroadcastShape(inputAShape);
    auto [broadcastShapeB, computeShapeB] = SplitBroadcastShape(inputBShape);

    auto broadcastShapeY = Shape::BroadcastOutputShape(broadcastShapeA, broadcastShapeB);
    auto computShapeY = ComputeShapeForLessTwoDim(computeShapeA, computeShapeB);

    auto outputShape = MergeShape(broadcastShapeY, computShapeY);

    OperandY()->MetaDataSameAs(inputA);
    OperandY()->SetShapeAndStride(outputShape);
}

PlacementSignature MatmulOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 2);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());

    const Shape& inputAShape = OperandA()->GetShape();
    const Shape& inputBShape = OperandB()->GetShape();
    const Shape& outputShape = OperandY()->GetShape();
    size_t outputDimSize = outputShape.NumAxis();
    auto [broadcastShapeA, computeShapeA] = SplitBroadcastShape(inputAShape);
    auto [broadcastShapeB, computeShapeB] = SplitBroadcastShape(inputBShape);

    // Broadcast shape part
    BroadcastBinaryOp::AddBroadcastBinaryOpPlacementSignature(builder, broadcastShapeA, broadcastShapeB);

    // --- Compute shape part ---
    // Placement rules for each dimension combination.
    // See file-header Placements 推导 section for notation, reference tables, and rule semantics.

    if (computeShapeA.NumAxis() == 1 && computeShapeB.NumAxis() == 1) {
        // 1D×1D dot product [K]×[K]→scalar: compute output has no dims, no placement rules needed.
        // Finish() adds Replicate→Replicate fallback for this case.
    } else if (computeShapeA.NumAxis() == 1 && computeShapeB.NumAxis() == 2) {
        // --- 1D×2D: [K] × [K, N] → [N] ---
        size_t kIdxA = inputAShape.NumAxis() - 1;  // A's K dim
        size_t kIdxB = inputBShape.NumAxis() - 2;  // B's K dim
        size_t nIdx = inputBShape.NumAxis() - 1;   // B's N dim
        size_t outNIdx = outputDimSize - 1;        // output N dim

        // Shard(N) on B → Shard(N) on Y
        builder.AddInput("R").AddInput(Shard(nIdx)).AddOutput(Shard(outNIdx)).Build();

        // K dim Shard: either side → Partial
        builder.AddInput(Shard(kIdxA)).AddInput("R").AddOutput("P").Build();
        builder.AddInput("R").AddInput(Shard(kIdxB)).AddOutput("P").Build();
        // K dim Shard: both sides → Partial
        builder.AddInput(Shard(kIdxA)).AddInput(Shard(kIdxB)).AddOutput("P").Build();

        // Edge: Shard(K) on A + Shard(N) on B → Partial (K conv constricts N propagation)
        builder.AddInput(Shard(kIdxA)).AddInput(Shard(nIdx)).AddOutput("P").Build();

        // Partial propagation
        builder.AddInput("P").AddInput("R").AddOutput("P").Build();
        builder.AddInput("R").AddInput("P").AddOutput("P").Build();
    } else if (computeShapeA.NumAxis() == 2 && computeShapeB.NumAxis() == 1) {
        // --- 2D×1D: [M, K] × [K] → [M] ---
        size_t mIdx = inputAShape.NumAxis() - 2;   // A's M dim
        size_t kIdxA = inputAShape.NumAxis() - 1;  // A's K dim
        size_t kIdxB = inputBShape.NumAxis() - 1;  // B's K dim (1D, only dim)
        size_t outMIdx = outputDimSize - 1;        // output M dim

        // Shard(M) on A → Shard(M) on Y
        builder.AddInput(Shard(mIdx)).AddInput("R").AddOutput(Shard(outMIdx)).Build();

        // K dim Shard: either side → Partial
        builder.AddInput(Shard(kIdxA)).AddInput("R").AddOutput("P").Build();
        builder.AddInput("R").AddInput(Shard(kIdxB)).AddOutput("P").Build();
        // K dim Shard: both sides → Partial
        builder.AddInput(Shard(kIdxA)).AddInput(Shard(kIdxB)).AddOutput("P").Build();

        // Edge: Shard(M) on A + Shard(K) on B → Partial (K conv constricts M propagation)
        builder.AddInput(Shard(mIdx)).AddInput(Shard(kIdxB)).AddOutput("P").Build();

        // Partial propagation
        builder.AddInput("P").AddInput("R").AddOutput("P").Build();
        builder.AddInput("R").AddInput("P").AddOutput("P").Build();
    } else if (computeShapeA.NumAxis() == 2 && computeShapeB.NumAxis() == 2) {
        // --- 2D×2D GEMM: [M, K] × [K, N] → [M, N] ---
        size_t mIdx = inputAShape.NumAxis() - 2;   // A's M dim
        size_t kIdxA = inputAShape.NumAxis() - 1;  // A's K dim
        size_t kIdxB = inputBShape.NumAxis() - 2;  // B's K dim
        size_t nIdx = inputBShape.NumAxis() - 1;   // B's N dim
        // Output indices are output-relative to account for batch-dim offset
        // when broadcast dims differ between inputs (e.g. A=[B,M,K], B=[K,N]).
        size_t outMIdx = outputDimSize - 2;  // output M dim
        size_t outNIdx = outputDimSize - 1;  // output N dim

        // Shard(M) on A, Replicate on B → Shard(M) on Y
        builder.AddInput(Shard(mIdx)).AddInput("R").AddOutput(Shard(outMIdx)).Build();
        // Replicate on A, Shard(N) on B → Shard(N) on Y
        builder.AddInput("R").AddInput(Shard(nIdx)).AddOutput(Shard(outNIdx)).Build();

        // K dim Shard: single-sided → Partial
        builder.AddInput(Shard(kIdxA)).AddInput("R").AddOutput("P").Build();
        builder.AddInput("R").AddInput(Shard(kIdxB)).AddOutput("P").Build();
        // K dim Shard: both sides → Partial
        builder.AddInput(Shard(kIdxA)).AddInput(Shard(kIdxB)).AddOutput("P").Build();

        // Edge: Shard(M) on A + Shard(K) on B → Partial
        builder.AddInput(Shard(mIdx)).AddInput(Shard(kIdxB)).AddOutput("P").Build();
        // Edge: Shard(K) on A + Shard(N) on B → Partial
        builder.AddInput(Shard(kIdxA)).AddInput(Shard(nIdx)).AddOutput("P").Build();

        // Partial propagation
        builder.AddInput("P").AddInput("R").AddOutput("P").Build();
        builder.AddInput("R").AddInput("P").AddOutput("P").Build();
    }

    return builder.Finish();
}

// FLOPs derivation for matmul / bmm with batched broadcast.
//
// Split each input into (broadcastDims, computeDims): computeDims are the trailing <= 2 axes (the
// matrix part) and broadcastDims are everything else (the batch part). The two batch parts are
// broadcast together into a single `batch` count — exactly the output's batch region.
//
// The matrix part maps the four torch.matmul rank cases onto M and N (K is the contraction axis):
//   2D×2D [M,K]×[K,N]  → M = rows of A,            N = cols of B
//   1D×2D [K]×[K,N]    → M = 1 (A is a row vec),   N = cols of B
//   2D×1D [M,K]×[K]    → M = rows of A,            N = 1 (B is a col vec)
//   1D×1D [K]×[K]      → M = N = 1 (dot product)
// so M is A's 0th compute axis when A is 2D (else 1), and N is B's 1st compute axis when B is 2D
// (else 1). Each of the (batch * M * N) output elements performs K multiply-accumulates; counting one
// multiply and one accumulate as 2 FLOPs (the standard MAC convention) gives 2 * batch * M * N * K.
OperatorCost MatmulOp::GetOperatorCost() const {
    DDebugAssert(GetInputSize() == 2);
    const Shape& shapeA = OperandA()->GetShape();
    const Shape& shapeB = OperandB()->GetShape();

    auto [broadcastShapeA, computeShapeA] = SplitBroadcastShape(shapeA);
    auto [broadcastShapeB, computeShapeB] = SplitBroadcastShape(shapeB);

    int64_t batch = static_cast<int64_t>(Shape::BroadcastOutputShape(broadcastShapeA, broadcastShapeB).Count());
    int64_t M = (computeShapeA.NumAxis() == 2) ? static_cast<int64_t>(computeShapeA[0]) : 1;
    int64_t K = static_cast<int64_t>(computeShapeA[computeShapeA.NumAxis() - 1]);
    int64_t N = (computeShapeB.NumAxis() == 2) ? static_cast<int64_t>(computeShapeB[1]) : 1;

    return OperatorCost::Compute(2 * batch * M * N * K);
}

}  // namespace core
}  // namespace dtorch
