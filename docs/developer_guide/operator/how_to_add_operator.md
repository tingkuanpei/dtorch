# How To Add Operator

新增算子需要完成五个层次的改造：**算子注释 → C++ 核心算子 → Kernel 计算 → C++/Python API 暴露 → 单元测试**。
其中**注释文档是第一步**，在编写任何 C++ 代码之前先在 `.cc` 文件中完成算子说明。

| 步骤 | 层 | 核心工作 | 关键文件 |
|---|---|---|---|
| 0 | 算子注释 | 在 `.cc` 文件中写功能描述、Shape 推导、Placements 规则 | `dtorch/core/operators/*/*.cc` |
| 1 | C++ 核心算子 | 定义 Param、Operator 类，重写虚函数 | `dtorch/core/operators/` |
| 2 | Kernel 计算 | 实现 `Compute()` 或自定义 Kernel | `dtorch/core/kernel/` |
| 3 | API 暴露 | 添加 C++ API → 添加 Python API | `dtorch/api/cpp/functional/`, `dtorch`, `dtorch.Tensor`, `dtorch.nn.functional` |
| 4 | 单元测试 | 以 PyTorch 为基准编写单元测试 | `python/dtorch/test/operators/` |

---

## 1. 编写算子注释文档

在编写任何 C++ 代码之前，**首先在算子的 `.cc` 文件中完成注释文档**。良好的注释帮助后续维护者快速理解算子的功能、Shape 计算逻辑和分布式行为。注释应覆盖以下三个核心部分。

### 1.1 算子功能描述

总结该算子在 [PyTorch 官方 API 文档](https://pytorch.org/docs/stable/torch.html) 中描述的功能，包括：

- **数学定义或操作语义**：算子的数学公式、操作含义（如 `torch.matmul` 的矩阵乘法语义、`torch.sum` 的规约语义）
- **支持的参数及其含义**：每个参数的类型、默认值、作用
- **边界条件和特殊行为**：如 1D 输入的处理、空张量的行为、Integer 类型的特殊处理、inplace 操作的管理

```cpp
// MatmulOp — Matrix Multiplication
//
// == 功能描述 (Functionality) ==
// 实现 torch.matmul / torch.mm / torch.bmm 的语义。
// - 2D × 2D: 标准矩阵乘法 [M, K] × [K, N] → [M, N]
// - ND × ND: 批量矩阵乘法，batch 维度支持 broadcast
// - 1D × 2D: 向量乘矩阵 [K] × [K, N] → [N]
// - 2D × 1D: 矩阵乘向量 [M, K] × [K] → [M]
// - 1D × 1D: 点积 [K] × [K] → scalar
```

### 1.2 输出 Tensor 的 Shape 计算

说明如何根据输入 Tensor 的 Shape 和算子参数计算输出 Tensor 的 Shape，包括：

- **Shape 计算公式或推导逻辑**：如 `ConvOp` 的输出 H/W 计算公式 `(input + pad - dilation*(kernel-1) - 1) / stride + 1`
- **广播（broadcast）规则**：哪些维度会广播，广播后的 Shape 如何确定
- **维度变化的处理**：如 reduce 移除维度（`keepdim=False`）、reshape 重组维度、transpose 交换维度顺序

```cpp
// == 输出 Shape (Output Shape) ==
// 输出 Shape 通过以下步骤推导：
// 1. 将每个输入 shape 拆分为 (batch_dims, compute_dims)：
//    - batch_dims = shape 去掉最后 2 维的部分
//    - compute_dims = shape 的最后 2 维（或最后 1 维，对 1D 输入）
// 2. 对 batch 维度执行广播（BroadcastOutputShape）。
// 3. 对 compute 维度执行矩阵乘法：
//    - [M, K] × [K, N] → [M, N]
//    - [K] × [K, N] → [N]  （1D × 2D）
//    - [M, K] × [K] → [M]  （2D × 1D）
//    - [K] × [K] → scalar  （1D × 1D）
// 4. 合并广播后的 batch shape 与 compute 输出 shape。
```

### 1.3 输出 Tensor 的 Placements 推导

说明输出 Tensor 的分布式 Placements 推导规则，即 **PlacementSignature** 的构建逻辑。PlacementSignature 描述了算子输入 Tensor 的 Placement 与输出 Tensor 的 Placement 之间的映射关系。当输入 Tensor 是 DTensor 时，框架根据此签名自动推导输出的分布式信息。

关于 PlacementSignature 的完整机制说明见 [PlacementSignature 文档](placement_signature.md)。

注释中应包含：

- **各维度上的 Shard/Replicate/Partial 映射**：输入维度的 Placement 如何映射到输出维度
- **何时产生 Partial**：当两个输入的 Shard 维度在规约维度上相交时，结果变为 Partial（需要后续 all-reduce 才能得到完整值）
- **Placement 参考表**：用 ASCII 表格清晰列出各输入维度 Placement 到输出 Placement 的映射关系
- **特殊情况的说明**：如 1D 输入、broadcast 维度、Optional 输入（bias）的处理

```cpp
// == Placements 推导 (PlacementSignature) ==
//
// 记号：对于 shape 为 [B..., M, K] 的输入 A 和 [B..., K, N] 的输入 B：
//   M_idx = inputAShape.NumAxis() - 2  (A 的行维度)
//   K_idx_A = inputAShape.NumAxis() - 1 (A 的规约维度)
//   K_idx_B = inputBShape.NumAxis() - 2 (B 的规约维度)
//   N_idx = inputBShape.NumAxis() - 1  (B 的列维度)
//
// GEMM PlacementSignature 参考表 (2D × 2D 及 broadcast 对齐情况):
// ---------------------
// |  A  |  B  |  Y  |
// ---------------------
// | S(M)|  R  | S(M)|  ← 行维度 Shard 从 A 传播到 Y
// |  R  | S(N)| S(N)|  ← 列维度 Shard 从 B 传播到 Y
// | S(K)| S(K)|  P  |  ← 规约维度两侧 Shard → Partial（需要 all-reduce）
// |  P  |  R  |  P  |  ← Partial 通过任意侧传播
// |  R  |  P  |  P  |
//
// 对于 1D × 2D（向量 × 矩阵）：A shape [K], B shape [K, N], 输出 [N]
//   - 仅 dim 0 参与匹配（匹配维度 = min(A_rank, B_rank) = 1）
//   - 在 dim 0：A 的 K 维 Placement + B 的 K 维 Placement → 输出 N 维 Placement
//   - 任一侧 Shard(K) 产生 Partial 输出（每个设备持有点积的部分结果）
//
// 对于 2D × 1D（矩阵 × 向量）：A shape [M, K], B shape [K], 输出 [M]
//   - 匹配 dim 0: A 的 M 维 + B 的 K 维 → 输出 M 维（从 A 传播 Shard(M)）
//   - 匹配 dim 1: A 的 K 维 + B 的 K 维 → 无输出 dim 1（输出为 [M]）
//   - 任一侧 Shard(K) 产生 Partial 输出
```

### 1.4 注释风格指南

| 部分 | 建议标题 | 关键内容 |
|---|---|---|
| 功能描述 | `== 功能描述 ==` 或 `== Functionality ==` | 数学定义、参数说明、边界条件 |
| Shape 计算 | `== 输出 Shape ==` 或 `== Output Shape ==` | 计算公式、广播规则、维度变化推导步骤 |
| Placements | `== Placements 推导 ==` 或 `== PlacementSignature ==` | Shard/Replicate/Partial 映射表、特殊情况说明 |

**核心原则**：

- 注释写在 **`.cc` 文件**中（不是 `.h` 文件）——实现细节是 `.cc` 的关注点
- 在 PlacementSignature 构建代码上方写 Placements 参考表，逐条解释每个 `builder.AddInput(...).AddOutput(...).Build()` 的含义
- 使用 ASCII 表格清晰展示维度映射关系
- 注释语言中英文皆可，关键术语（Shard、Replicate、Partial、PlacementSignature）保留英文

### 1.5 参考示例

以下文件中包含规范的算子注释，可作为参考：

| 文件 | 注释亮点 |
|---|---|
| `dtorch/core/operators/standard/matmul_op.cc` | ★ 最佳实践：完整的功能描述 + Shape 推导步骤 + PlacementSignature ASCII 参考表 + 各 case 逐条规则注释 |
| `dtorch/core/operators/standard/linear_op.cc` | GEMM/Linear 的双重 Placement 参考表（GEMM 表 + Linear 含 bias OptionalInput 的映射表） |
| `dtorch/core/operators/standard/reduce_op.cc` | Reduce 类算子的 Shard→Replicate 逐维映射逻辑（`keepdim` 处理 + 全规约 vs 部分规约） |
| `dtorch/core/operators/standard/broadcast_binary_op.cc` | 广播维度对齐的 Shard 传播规则（独有维度 + 共享维度） |

---

## 2. 完善或新增 C++ 核心算子

> **前置步骤**：在编写任何 C++ 代码前，请先在 `.cc` 文件中完成 [第 1 章](#1-编写算子注释文档) 所述的注释文档（功能描述、Shape 推导、Placements 规则）。

C++ 核心算子采用**模板方法模式（Template Method Pattern）**，抽象基类为 `dtorch::core::Operator`（定义在 `dtorch/core/operators/operator.h`）。派生类按需重写父类的虚函数，完整说明见 [Operator 模板方法模式文档](operator_template_method_pattern.md)。

### 2.1 判断：新建算子还是复用现有算子

DTorch 已经实现了大量算子。如果新增的算子与现有算子属于同类（即输出 Tensor 的 Shape 推断逻辑、分布式规则一致），应**复用现有算子**，通过 `enum` 区分子类型。

**示例**：`relu`、`sigmoid`、`gelu`、`silu` 等激活函数，输出 Shape 与输入完全一致，统一归为 `dtorch/core/operators/standard/activation_op.h` 中的 `ActivationOp`，通过 `enum class ActivationType` 进行区分。

现有算子按功能分为三类目录：

| 目录 | 用途 | 示例 |
|---|---|---|
| `dtorch/core/operators/standard/` | 常规算子，基于 LibTorch 后端 | `ActivationOp`, `BroadcastBinaryOp`, `ConvOp`, `MatmulOp` |
| `dtorch/core/operators/fused_compile/` | 融合编译算子，调用 `torch.compile` 执行 | `ApplyRotaryEmbOp`, `SiluLinearChunkOp` |
| `dtorch/core/operators/system/` | 系统算子，实现框架调度相关功能 | `NvtxOp`（性能标记）, `MemoryOp`（显存管理） |

此外还有两个顶层特殊算子：`FuseOp`（子图融合节点）和 `SubGraphOp`（子图封装）。

DTorch 所有已有算子及其映射关系参见 [operators_mapping.md](operators_mapping.md)。**如果新增算子，需同步更新该文档。**

### 2.2 新增 OperatorType

在 `dtorch/core/operators/operator_param.h` 的 `DTORCH_FOREACH_OPERATOR_TYPE` 宏中新增一项：

```cpp
#define DTORCH_FOREACH_OPERATOR_TYPE(Func)              \
    Func(Activation                     ,  0)           \
    Func(BatchNorm                      ,  1)           \
    // ... 已有算子 ...
    Func(Clone                          , 46)           \
    Func(YourNewOp                      , 47)           \   // ← 新增：递增 ID
```

此宏通过 X-Macro 技术自动生成 `enum class OperatorType` 的枚举值 `kYourNewOp`，并驱动 `OpTypeToString()` / `OpTypeFromString()` 等工具函数。

### 2.3 定义 Param 类

每个算子需定义同名的参数结构体，继承自 `OpParam`，用于存储算子参数并支持序列化。

**无参数的算子**可使用 `NoElementOpParam` 模板：

```cpp
using ContiguousParam = NoElementOpParam<OperatorType::kContiguous>;
```

**有参数的算子**需自定义 Param 结构体：

```cpp
// dtorch/core/operators/standard/activation_op.h
struct ActivationParam : public OpParam {
    ActivationType activationType;
    bool inplace;
    double alpha;
    double beta;
    std::string approximate;

public:
    // 构造函数：必须在初始化列表中调用 OpParam(OperatorType::kActivation)
    ActivationParam(ActivationType activationType = ActivationType::kReLU,
                    bool inplace = false, double alpha = 0.0,
                    double beta = 0.0, const std::string& approximate = "")
        : OpParam(OperatorType::kActivation),
          activationType(activationType), inplace(inplace),
          alpha(alpha), beta(beta), approximate(approximate) {}

    // 序列化：必须序列化基类 + 所有成员
    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & BaseObject<OpParam>(*this);   // 必须：基类序列化（包含 mOpType）
        ar & activationType;
        ar & inplace;
        ar & alpha;
        ar & beta;
        ar & approximate;
    }
};
```

**关键规则**：
- 构造函数必须在初始化列表中调用 `OpParam(对应OperatorType)`，确保 `mOpType` 正确设置。
- `serialize()` 必须以 `ar & BaseObject<OpParam>(*this);` 开头。
- 所有参数类型应为 `float`/`double` 或 `int32_t`，需与 ONNX 定义保持一致（参考 [ONNX Operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md)）。

### 2.4 实现 Operator 类并重写虚函数

派生类继承 `Operator`，按需重写虚函数。**所有可重写函数的完整指南见 [Operator 模板方法模式文档](operator_template_method_pattern.md)**。

**最小实现模板**（以 ActivationOp 为例）：

```cpp
// dtorch/core/operators/standard/activation_op.h
class ActivationOp : public Operator {
public:
    ActivationOp(std::shared_ptr<OpParam> opParam) : Operator(opParam) {}

    // ★ 必须重写：推断输出 Tensor 的 Shape/Stride/DataKind
    void InferOutputMetaInfo() const override;

    // 可选：跳过分布式推断（Element-wise 算子输出 Placement 与输入一致）
    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    // 可选：实现实际计算（也可在 Kernel 中实现）
    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override;

    // ★ 必须重写：返回算子描述字符串，用于调试和日志输出
    std::string GetDescribeString() const override;
};
```

**重写优先级（从必须到可选）**：

| 优先级 | 虚函数 | 说明 |
|---|---|---|
| ★ 必须 | `InferOutputMetaInfo()` | 纯虚函数，推断输出 Shape/Stride/DataKind |
| ★ 必须 | `GetDescribeString()` | 返回算子描述字符串，用于调试、日志和 Graph 可视化。**每个派生类都应重写**，至少包含算子名和输入输出 Shape |
| 高 | `Compute()` | 绝大多数算子需重写，调用 LibTorch 执行实际计算 |
| 高 | `GetPlacementSignature()` | 分布式算子需定义输入→输出 Placement 映射规则，详见 [PlacementSignature 文档](placement_signature.md)。**编写前应先在 `.cc` 文件中写好 Placements 参考表注释（参见第 1 章）** |
| 中 | `InferOutputSize()` | 默认返回 1（单个输出）。多输出算子（如 `ChunkOp`）或零输出算子（如 `NvtxOp`）需重写 |
| 中 | `SkipDistributedSpecFromPlacementSignature()` | Element-wise 算子或无非分布式输入时返回 `true` |
| 低 | `IsRequireInputSameDataKind()` | 默认 `true`。混合数据类型的算子（如 `WhereOp`）重写返回 `false` |
| 低 | `InferOperatorAssignInfo()` | 默认根据输入输出 Operand 的 DeviceMesh 自动分配。无输入输出的算子（如 `NvtxOp`）需重写 |
| 低 | `CheckInput()` | 基类已做了通用校验，派生类可添加额外的输入校验逻辑 |
| 低（算力密集算子必做） | `GetOperatorCost()` | 代价估算（FLOPs / 带宽）。基类默认按带宽密集估算（= 输入+输出字节之和），**算力密集算子（matmul、conv、sdpa、linear、outer 等）必须覆写**返回 `OperatorCost::Compute(flops)`，其余算子继承默认即可。详见 [算子代价估算 (OperatorCost)](operator_cost.md) |

### 2.5 序列化注册

在 `dtorch/core/operators/operator_serialization_pack.h` 中完成两项修改：

**a) 添加头文件 include**（在文件顶部）：

```cpp
#include "standard/your_new_op.h"   // ← 新增
```

**b) 确认序列化模板函数**已通过 `DTORCH_FOREACH_OPERATOR_TYPE` 宏自动展开覆盖新增类型。该文件中的 `OperatorSerializationPack::serialize()` 模板函数使用宏展开生成所有算子类型的序列化代码，因此**只要 `DTORCH_FOREACH_OPERATOR_TYPE` 宏中包含了新算子，此处自动生效**，无需手动修改 `switch-case`。

### 2.6 工厂注册

在 `dtorch/core/operators/operator_factory.cc` 中完成两项修改：

**a) 添加头文件 include**（在文件顶部）：

```cpp
#include "standard/your_new_op.h"   // ← 新增
```

**b) 在 `OperatorFactory::OperatorFactory()` 构造函数中添加注册调用**：

```cpp
OperatorFactory::OperatorFactory() : mOpConstructorMap() {
    RegisterOpConstructor<ActivationOp>(OperatorType::kActivation);
    // ... 已有注册 ...
    RegisterOpConstructor<YourNewOp>(OperatorType::kYourNewOp);   // ← 新增
}
```

`RegisterOpConstructor<T>(OperatorType)` 建立 `OperatorType` 到构造函数的映射。运行时通过 `NewOperatorOrThrow(param, inputs)` 创建算子实例，完成 `Infer()` 和拓扑注册。

### 2.7 步骤 2 检查清单

- [ ] 在 `.cc` 文件中完成算子注释文档（功能描述、Shape 推导、Placements 规则，参见第 1 章）
- [ ] 判断创建新算子还是复用现有算子
- [ ] 在 `operator_param.h` 的 `DTORCH_FOREACH_OPERATOR_TYPE` 宏中增加 `OperatorType`
- [ ] 实现 Param 类（继承 `OpParam`，构造函数传入正确的 `OperatorType`，实现 `serialize()`）
- [ ] 实现 Operator 类（继承 `Operator`，至少实现 `InferOutputMetaInfo()`）
- [ ] **如为算力密集算子（matmul/conv/sdpa/linear/outer 等），覆写 `GetOperatorCost()` 返回 FLOPs**（在 `Infer()` 之后调用，直接读取已推断的 Operand Shape，详见 [算子代价估算 (OperatorCost)](operator_cost.md)）
- [ ] 在 `operator_serialization_pack.h` 中添加头文件 include
- [ ] 在 `operator_factory.cc` 中添加头文件 include + `RegisterOpConstructor` 注册
- [ ] 更新 `operators_mapping.md` 文档

---

## 3. 完善或新增对应的 Kernel

算子的实际计算由 `Operator::Compute()` 函数或自定义 Kernel 完成。两者按复杂度选择：

### 3.1 简单方案：重写 `Operator::Compute()`

**适用场景**：计算逻辑仅需调用 LibTorch 接口，不需要获取 CUDA stream、ThreadGroup 等额外上下文。

`dtorch/core/kernel/kernel.cc` 中的 `Kernel::Compute()` 默认调用 `Operator::Compute()`，因此子类直接重写 `Compute()` 即可：

```cpp
// dtorch/core/operators/standard/activation_op.cc
void ActivationOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<ActivationParam>();
    switch (param.activationType) {
        case ActivationType::kReLU:
            outputs[0] = torch::relu(inputs[0].value());
            break;
        case ActivationType::kSigmoid:
            outputs[0] = torch::sigmoid(inputs[0].value());
            break;
        // ...
    }
}
```

**代码位置选择**：
- 如果 `Compute()` 实现简洁（一个函数内完成），建议实现在 `dtorch/core/kernel/torch_kernel.cc` 中，以减少编译时 include Torch 头文件的次数，提高编译速度。
- 如果实现较复杂，可在对应算子的 `.cc` 文件中实现（如 `dtorch/core/operators/standard/broadcast_binary_op.cc` 中的 `BroadcastBinaryOp::Compute`）。

### 3.2 复杂方案：新增自定义 Kernel

**适用场景**：计算需要获取 CUDA stream、ThreadGroup、或者需要自定义的同步/调度逻辑。

此时在 `dtorch/core/kernel/kernel_implement/` 目录新增 Kernel 类，并通过 `KernelFactory` 完成注册。自定义 Kernel 可访问 `dtorch/core/kernel/kernel.h` 提供的完整接口（Stream 管理、Event 同步等）。

#### 3.2.1 Kernel 类模板

```cpp
// dtorch/core/kernel/kernel_implement/my_custom_kernel.h
#pragma once

#include "dtorch/core/kernel/kernel.h"

namespace dtorch {
namespace core {

class MyCustomKernel : public Kernel {
public:
    MyCustomKernel(const KernelCreateCtx& ctx) : Kernel(ctx) {}
    DTORCH_DEFAULT_COPY_AND_MOVE(MyCustomKernel);

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) override;
};

}  // namespace core
}  // namespace dtorch
```

#### 3.2.2 工厂注册

在 `dtorch/core/kernel/kernel_factory.cc` 中完成两项修改：

**a) 添加头文件 include**（在文件顶部）：

```cpp
#include "kernel_implement/my_custom_kernel.h"   // ← 新增
```

**b) 在 `KernelFactory::KernelFactory()` 构造函数中添加注册调用**：

```cpp
KernelFactory::KernelFactory() : mKernelConstructorMap() {
    RegisterKernelConstructor<ConvertKernel>(OperatorType::kConvert);
    // ... 已有注册 ...
    RegisterKernelConstructor<MyCustomKernel>(OperatorType::kMyOpType);   // ← 新增
}
```

`RegisterKernelConstructor<KernelClass>(OperatorType)` 将 `OperatorType` 映射到 Kernel 构造函数 lambda。运行时 `KernelFactory::NewKernel(ctx)` 根据 `ctx.op->GetOpType()` 查找并创建对应的 Kernel 实例；若未找到匹配项，fallback 到基类 `Kernel`（调用 `mOp->Compute()`）。

一个 Kernel 类可以注册到多个 `OperatorType`（如 `ViewKernel` 同时注册 `kView` 和 `kReshape`）。

**已有自定义 Kernel 的 Operator**：`CopyOp`（`CopyKernel`）、`ConvertOp`（`ConvertKernel`）、`CreateOp`（`CreateKernel`）、`ViewOp`/`ReshapeOp`（`ViewKernel`）、`ReduceOp`（`ReduceKernel`）、`MemoryOp`（`MemoryKernel`）、`SyncOp`（`SyncKernel`）。

---

## 4. 增加 C++ API 和 Python API

### 4.1 C++ API

在 `dtorch/api/cpp/functional/` 目录添加 C++ API。文件按功能分组，如 `activation.h/.cc`、`math.h/.cc`、`neural_network.h/.cc` 等。

**头文件** (`dtorch/api/cpp/functional/your_op.h`)：

```cpp
#pragma once
#include "../tensor.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor YourFunction(const Tensor& input, double param1, bool param2 = false);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
```

**实现文件** (`dtorch/api/cpp/functional/your_op.cc`)：

```cpp
#include "your_op.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/standard/your_new_op.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor YourFunction(const Tensor& input, double param1, bool param2) {
    std::unique_ptr<core::OpParam> param(new core::YourNewParam(/* ... */));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
```

核心模式：**构造 `OpParam` → 调用 `GraphConstructor::AddOperator()` 将算子添加到计算图中**。`AddOperator` 内部调用 `OperatorFactory::NewOperatorOrThrow()` 完成算子创建和推断。

### 4.2 Python 绑定（自动生成）

C++ API 会通过 nanobind 自动绑定到 `dtorch._dtorch_py_api.nn.functional` 模块。绑定代码位于 `dtorch/api/python/` 目录，包括：
- `nanobind_register.cc` — 总注册入口
- `functional/py_bind_functional_generated.h` — 自动生成的函数绑定
- `functional/py_bind_functional_manual.h` — 手动编写的函数绑定

**新添加的 C++ API 函数需要在此增加对应的 nanobind 绑定代码**，绑定过程中接口名会做大小写转换以符合 Python 命名规范（如 `Relu` → `relu`）。

### 4.3 Python API

Python 层通过 `python/dtorch/nn/functional.py` 中的 `convert_cpp_instance_decorator` 装饰器自动从 `dtorch._dtorch_py_api.nn.functional` 导入所有 C++ 函数并包装为 Python 接口。

**命名规则**：
- 以 `_` 开头的 C++ 函数 → 去掉前导下划线后导入 `dtorch` 模块（如 `_add` → `dtorch.add`）
- 不以 `_` 开头的 C++ 函数 → 导入 `dtorch.nn.functional` 模块（如 `conv2d` → `dtorch.nn.functional.conv2d`）

**API 设计原则**：DTorch 的 Python 接口**必须与 PyTorch 保持一致**（参考 [PyTorch API 文档](https://docs.pytorch.org/docs/2.12/pytorch-api.html)）。涉及三种调用位置：
- `dtorch.xxx` — 顶层函数（如 `dtorch.add`, `dtorch.matmul`）
- `dtorch.Tensor.xxx` — Tensor 方法（如 `tensor.clone()`）
- `dtorch.nn.functional.xxx` — 函数式 API（如 `dtorch.nn.functional.conv2d`）

各层之间的映射关系参见 [operators_mapping.md](operators_mapping.md)。

---

## 5. 完善或新增 Python 单元测试

### 5.1 测试文件结构

在 `python/dtorch/test/operators/` 目录增加对应的单元测试文件，命名规范为 `test_<operator>.py`。

### 5.2 测试模板

测试必须使用 **PyTorch 接口作为测试基准**。推荐使用 `OrderedDict` + `gen_arg_list` 的组合进行参数化测试。

**必须覆盖所有执行路径**，确保代码的每个分支都经过验证：

- **Shape 计算逻辑**：覆盖不同维度的输入（1D、2D、3D、4D、ND）、不同大小的 Shape（含边界值如 0、1、大尺寸）、广播场景（不同输入的维度不一致时）
- **Placements 推导逻辑**：覆盖 DTensor 上所有合法的 Placement 组合（Shard(dim)、Replicate、Partial 的各种排列），验证每种 PlacementSignature 规则都能正确命中
- **输入 Tensor 组合**：覆盖所有输入组合（如 binary op 的两个输入分别来自不同的 DeviceMesh 场景、Optional 输入有无的情况）
- **不同 Device**：CPU + CUDA
- **不同 DataKind / DType**：float32、float16、int64、bool 等
- **边界条件**：空 Tensor、标量 Tensor、极端值（NaN、Inf）、inplace 操作
- **非分布式 + 分布式**：先验证非分布式 Tensor 的结果正确性（与 PyTorch 对齐），再验证分布式 Tensor 的结果正确性

```python
"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Your Name (contact: your@email.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_allclose, assert_tensor_equal


def _test_your_op(test_case, shape, device, dtype):
    # 1. 构造输入：PyTorch 和 DTorch 使用相同的数据
    torch_in = torch.rand(*shape, device=device, dtype=dtype)
    dtorch_in = dtorch.Tensor(torch_in.clone())

    # 2. 可选：构造分布式 Tensor
    device_mesh = dtorch.DeviceMesh(device, range(shape[0]))
    placements = [dtorch.Shard(0)]
    dtorch_d_in = dtorch.Tensor(torch_in.clone(), device_mesh=device_mesh, placements=placements)

    # 3. 非分布式测试：必须使用 assert_tensor_equal 精确比较
    torch_out = torch.nn.functional.your_op(torch_in, param1=0.5)
    dtorch_out = dtorch.nn.functional.your_op(dtorch_in, param1=0.5)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    # 4. 分布式测试：使用 assert_tensor_allclose 容忍集合通信浮点误差
    dtorch_out = dtorch.nn.functional.your_op(dtorch_d_in, param1=0.5)
    assert_tensor_allclose(test_case, torch_out, dtorch_out)


class TestYourOp(unittest.TestCase):
    def test_your_op(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_your_op(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
```

### 5.3 运行测试

```bash
# 运行单个算子测试
python3 python/dtorch/test/operators/test_your_op.py

# 运行 DTorch vs PyTorch 算子一致性测试（全量）
python3 python/dtorch/test/operators/test_functioanl.py
```

---

## 完整示例：新增一个简单算子

以下以新增 `MyClip` 算子为例，展示完整流程。

### Step 1: 在 `.cc` 文件中编写注释文档

在开始编写代码之前，先在 `my_clip_op.cc` 中写好注释：

```cpp
// MyClipOp — Clamp / Clip
//
// == 功能描述 ==
// 实现 torch.clamp / torch.clip 的语义。
// 将输入 Tensor 的所有元素限制在 [min_val, max_val] 范围内：
//   y_i = min(max(x_i, min_val), max_val)
// - 如果 min_val > max_val，行为未定义
// - 支持标量 min_val 和 max_val 参数
//
// == 输出 Shape ==
// Element-wise 算子，输出 Shape 与输入 Shape 完全一致。
//
// == Placements 推导 ==
// Element-wise 算子，输出 Placement 与输入完全相同，不需要 PlacementSignature。
// 通过 SkipDistributedSpecFromPlacementSignature() 返回 true 跳过分布式推断。
```

### Step 2: 在 `operator_param.h` 增加 OperatorType

```cpp
#define DTORCH_FOREACH_OPERATOR_TYPE(Func)              \
    // ... 已有 ...
    Func(Clone                          , 46)           \
    Func(MyClip                         , 47)           \
```

### Step 3: 创建 `dtorch/core/operators/standard/my_clip_op.h`

```cpp
#pragma once
#include "../operator.h"

namespace dtorch {
namespace core {

struct MyClipParam : public OpParam {
    double min_val;
    double max_val;

    MyClipParam(double min_val = 0.0, double max_val = 1.0)
        : OpParam(OperatorType::kMyClip), min_val(min_val), max_val(max_val) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & BaseObject<OpParam>(*this);
        ar & min_val;
        ar & max_val;
    }
};

class MyClipOp : public Operator {
public:
    MyClipOp(std::shared_ptr<OpParam> opParam) : Operator(opParam) {}

    // Element-wise 算子：输出 Shape 与输入相同
    void InferOutputMetaInfo() const override {
        OperandY()->SetShapeAndStrideV2(OperandX()->GetShape(), OperandX()->GetStride());
        OperandY()->SetDataKind(OperandX()->GetDataKind());
    }

    // Element-wise 算子：跳过分布式推断
    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const override {
        const auto& param = GetOpParam<MyClipParam>();
        outputs[0] = torch::clamp(inputs[0].value(), param.min_val, param.max_val);
    }
};

}  // namespace core
}  // namespace dtorch
```

### Step 4: 序列化注册

在 `operator_serialization_pack.h` 顶部添加：

```cpp
#include "standard/my_clip_op.h"
```

### Step 5: 工厂注册

在 `operator_factory.cc` 中：
- 顶部添加 `#include "standard/my_clip_op.h"`
- 构造函数中添加 `RegisterOpConstructor<MyClipOp>(OperatorType::kMyClip);`

### Step 6: C++ API

创建 `dtorch/api/cpp/functional/my_clip.h`：

```cpp
#pragma once
#include "../tensor.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor MyClip(const Tensor& input, double min_val, double max_val);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
```

创建 `dtorch/api/cpp/functional/my_clip.cc`：

```cpp
#include "my_clip.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/standard/my_clip_op.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor MyClip(const Tensor& input, double min_val, double max_val) {
    std::unique_ptr<core::OpParam> param(new core::MyClipParam(min_val, max_val));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
```

### Step 7: Python 绑定 + 测试

- 在 nanobind 绑定文件中注册 `MyClip` 函数
- 在 `python/dtorch/test/operators/` 创建 `test_my_clip.py`，以 `torch.clamp` 为基准编写测试
- 更新 `operators_mapping.md`

---

## 参考文档索引

| 文档 | 内容 |
|---|---|
| [Operator 模板方法模式](operator_template_method_pattern.md) | 所有可重写虚函数的完整说明、算法骨架、代码示例 |
| [PlacementSignature 文档](placement_signature.md) | 分布式规则定义、Builder API、典型实现模式 |
| [Operator 类说明](operators_class.md) | Operator 基类结构、派生类体系、生命周期 |
| [Operators Mapping](operators_mapping.md) | Python ↔ C++ API ↔ C++ 核心算子的映射关系 |
| [算子代价估算 (OperatorCost)](operator_cost.md) | `OperatorCost` 类、FLOPs / 带宽估算、`GetOperatorCost()` 覆写约定与公式推导 |
