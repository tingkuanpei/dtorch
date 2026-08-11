/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "activation_op.h"

#include <stdexcept>

// ActivationOp — Activation Functions (Element-wise)
//
// == 功能描述 (Functionality) ==
// 实现 PyTorch 激活函数的语义，统一封装为 ActivationOp，通过 ActivationType enum 区分子类型。
//
// 支持 6 种激活函数：
//
// | ActivationType   | PyTorch API             | 数学定义                                   |
// |------------------|-------------------------|--------------------------------------------|
// | kReLU            | torch.relu / F.relu     | y = max(0, x)                              |
// | kSigmoid         | torch.sigmoid           | y = 1 / (1 + exp(-x))                      |
// | kLeakyRelu       | F.leaky_relu            | y = x if x >= 0 else negative_slope * x    |
// | kELU             | F.elu                   | y = x if x >= 0 else alpha * (exp(x) - 1)  |
// | kGELU            | F.gelu                  | y = x * Phi(x), 支持 approximate="tanh"    |
// | kSiLU            | F.silu / torch.silu     | y = x * sigmoid(x)                         |
//
// 参数说明：
// - inplace: 是否就地修改输入 Tensor（默认 false）。ReLU/Sigmoid 不支持 inplace 时仍需创建新输出。
// - alpha: LeakyRelu 的 negative_slope（默认 0.01），ELU 的 scale（默认 1.0）
// - approximate: GELU 近似模式，"none"（精确）或 "tanh"（近似），默认 "none"
//
// 边界条件：
// - Integer 类型输入：仅 ReLU 支持（其余激活对整数输入无意义，抛出 std::invalid_argument）
// - 空 Tensor：输出 Shape 与输入一致，均为空
// - Scalar Tensor (0D)：输出同样为 scalar
//
// == 输出 Shape (Output Shape) ==
// Element-wise 算子，输出 Tensor 的 Shape、Stride、DataKind 与输入完全一致。
// InferOutputMetaInfo() 调用 OperandY()->MetaDataSameAs(OperandX()) 完成推导。
//
// == Placements 推导 (PlacementSignature) ==
// Element-wise 算子：每个输出元素仅依赖同位置的输入元素，不存在跨设备的数据依赖。
// 因此输出 Placement 与输入 Placement 完全一致，无需定义 PlacementSignature。
//
// 通过 SkipDistributedSpecFromPlacementSignature() 返回 true 跳过分布式推断，
// 框架自动将输入的 Placements 传播到输出。
//
// 对于任意输入 Placement，输出 Placement 保持相同（Partial 除外，见下方约束）：
//   Input Placement  →  Output Placement
//   Shard(0)         →  Shard(0)
//   Shard(N)         →  Shard(N)
//   Replicate()      →  Replicate()
//
// ★ Partial() 输入不支持：激活函数均为非线性运算（f(sum(x_i)) ≠ sum(f(x_i))），
//   Partial 输入在数学上没有意义。CheckInput() 中会显式拒绝 Partial 输入，
//   用户需先调用 redistribute(Replicate()) 再传入激活算子。

namespace dtorch {
namespace core {

void ActivationOp::CheckInput() const {
    Operator::CheckInput();

    DDebugAssert(GetInputSize() == 1 && GetOutputSize() == 1);

    // Non-linear activation functions do not commute with sum:
    // f(sum(x_i)) ≠ sum(f(x_i)), so Partial() input is mathematically invalid.
    if (OperandX()->IsDistributed() && OperandX()->GetPlacementSeq().HasPartial()) {
        throw std::invalid_argument(
            "ActivationOp: Partial() input is not supported for non-linear activation functions. "
            "Please redistribute the input to Replicate() or Shard() before applying activations.");
    }

    const auto& param = GetOpParam<ActivationParam>();

    // Integer type input is only valid for ReLU
    if (DataKindIsInteger(OperandX()->GetDataKind()) && param.activationType != ActivationType::kReLU) {
        throw std::invalid_argument("Input tensor's data type is integer, activation type isn't ReLU");
    }

    // GELU: validate approximate parameter
    if (param.activationType == ActivationType::kGELU) {
        if (!param.approximate.empty() && param.approximate != "tanh" && param.approximate != "none") {
            throw std::invalid_argument("approximate argument must be either none or tanh");
        }
    }
}

void ActivationOp::InferOutputMetaInfo() const { OperandY()->MetaDataSameAs(OperandX()); }

}  // namespace core
}  // namespace dtorch
