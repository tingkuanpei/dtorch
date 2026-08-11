# OperatorCost（算子代价估算）

`OperatorCost` 描述一个 `Operator` 在**当前输入/输出 shape 下**所需的**算力（FLOPs）**或**带宽（memory traffic, bytes）**。它用于后续计算每个算子、每段计算图的算力/带宽利用率（roofline 分析），是性能分析与优化（计算-通信重叠、算子调度、设备选型）的基础数据。

## 1. 核心概念

`OperatorCost` 是一个独立类（`dtorch/core/operators/operator_cost.h`），携带两个**正交**的"代价通道"，**通常只有一个通道有值，另一个为空**（`std::nullopt`）。二者并不互斥：少数既非纯算力密集、也非纯带宽密集的算子（如需要同时给出算力与带宽两个数值的融合 kernel）可经 `ComputeAndMemory()` 同时设置两个通道。

| 通道 | 字段 | 语义 | 典型算子 |
|---|---|---|---|
| 算力密集 | `Flops()` | 算术运算量（FLOPs） | matmul、conv、sdpa、linear、outer |
| 带宽密集 | `BandwidthBytes()` | 内存搬运量（字节） | add、copy、reshape、activation、layer_norm … |
| 混合 | 两者 | 算力 + 带宽同时报告 | 需完整 roofline 视角的融合算子（经 `ComputeAndMemory()`） |

判定接口：

| 接口 | 含义 |
|---|---|
| `IsComputeBound()` | 是否报告了 FLOPs（即 `Flops().has_value()`） |
| `IsMemoryBound()` | 是否报告了带宽（即 `BandwidthBytes().has_value()`） |
| `HasCost()` | 两个通道至少一个有值；系统/控制算子二者皆空时返回 `false` |

> `IsComputeBound()` 与 `IsMemoryBound()` 对同一对象**可同时为 `true`**（混合算子）。

```cpp
// dtorch/core/operators/operator_cost.h
class OperatorCost {
public:
    OperatorCost() = default;

    // 工厂函数：优先使用它们构造，而不是直接触碰私有字段
    static OperatorCost Compute(int64_t flops);                       // 算力密集
    static OperatorCost Memory(int64_t bandwidthBytes);               // 带宽密集
    static OperatorCost ComputeAndMemory(int64_t flops,               // 混合：算力 + 带宽同时报告
                                         int64_t bandwidthBytes);
    static OperatorCost FromOperands(const OperandArray& inputs,      // 带宽密集默认实现
                                     const OperandArray& outputs);

    bool IsComputeBound() const;
    bool IsMemoryBound() const;
    bool HasCost() const;
    const std::optional<int64_t>& Flops() const;
    const std::optional<int64_t>& BandwidthBytes() const;

    std::string ToString() const;  // 例: "OperatorCost[flops=123456]" / "OperatorCost[flops=N, bandwidthBytes=M]"
};
```

代价构造逻辑集中在 `OperatorCost` 内（`Compute` / `Memory` / `ComputeAndMemory` / `FromOperands`），算子侧只需选择工厂并喂入数值，保持算子实现精简。

## 2. 前置条件（重要）

> **`GetOperatorCost()` 在 `Infer()` 之后调用**：此时所有输入与输出 operand 的 shape 都已推导完毕、可直接读取。

因此覆写 `GetOperatorCost()` 时**应直接读取 operand 的 shape，而不是重新推导输出 shape**（不要重复 `InferOutputMetaInfo()` 里的 padding / 广播 / 输出尺寸计算逻辑）。例如卷积直接读输出 operand 的 `[outN, outC, outH, outW]`，而不是再用 `CalculateConvOutput()` 算一遍。

## 3. 基类默认实现（带宽密集）

`Operator::GetOperatorCost()` 提供带宽密集的默认实现：把所有**非 null** 的输入与输出 operand 的字节数求和（元素数 × 元素大小）。绝大多数 element-wise / 数据搬运算子（add、mul、relu、copy、reshape、layer_norm、softmax、concat、embedding …）继承该默认即可，无需手写。

```cpp
// dtorch/core/operators/operator.cc
OperatorCost Operator::GetOperatorCost() const {
    return OperatorCost::FromOperands(GetInputOperands(), GetOutputOperands());
}
```

`FromOperands` 会跳过 null-shape operand，因此缺失的可选输入（如 `null` bias）贡献 0 字节，而不会计入一个错误的 `(-100 × elementSize)` 项。

## 4. 算力密集算子覆写

对算力密集算子，在子类覆写 `GetOperatorCost()` 并返回 `OperatorCost::Compute(flops)`。当前已覆写的算子及其 FLOPs 公式（推导详见各算子 `.cc` 中的注释）：

| 算子 | 文件 | FLOPs 公式 |
|---|---|---|
| **MatmulOp** | `standard/matmul_op.{h,cc}` | `2 * batch * M * N * K` |
| **LinearOp** | `standard/linear_op.{h,cc}` | `2 * batchSize * inFeatures * outFeatures` |
| **ConvOp** | `standard/conv_op.{h,cc}` | `2 * outN * outC * outH * outW * inCPerGroup * kH * kW` |
| **SdpaOp** | `standard/scaled_dot_product_attention_op.{h,cc}` | `2 * batch * heads * seqQ * seqKv * (headDim + E_v)` |
| **OuterOp** | `standard/outer_op.{h,cc}` | `numelA * numelB`（见下方注 1） |

### 4.1 关于 `2×` 系数（MAC 约定）

matmul / linear / conv / sdpa 都在一个收缩维（contraction dim）上做乘加：每个输出元素执行 `K`（或 `inCPerGroup*kH*kW` 等）次 **multiply-accumulate**。按业界惯例把 1 次乘 + 1 次加记为 **2 FLOPs**，因此公式带前置系数 `2`。

### 4.2 注释要点

- **OuterOp 用 1× 而非 2×**：外积每个输出元素仅 1 次乘法、无累加（不存在收缩维），故 `numelA * numelB`，不加 2。
- **ConvOp 直接读输出 shape**：`outN/outC/outH/outW` 来自输出 operand（NCHW 与 NHWC 的通道位置随 `param.format` 区分）；`kH/kW` 来自 `param.kernelSize`；`inCPerGroup = inC / group`。不再重新推导输出空间尺寸。
- **SdpaOp 与 GQA**：K/V 头数少于 Q，但会复制/广播到 Q 头数，故 FLOPs 按 Q 头数（`qHeads`）缩放，无需因 `enableGqa` 分支。`headDim == kShape[-1]`（Q/K 收缩维），`E_v == vShape[-1]`（V 收缩维，可与 `headDim` 不同）。softmax/exp 的开销相对两个 GEMM 是带宽密集，略去不计。
- **LinearOp 的 bias 不计入**：尾部的 bias add 是 element-wise（带宽密集），不计入 GEMM 的 FLOPs。

### 4.3 混合算子（同时报告算力与带宽）

少数既非纯算力密集、也非纯带宽密集的算子（典型是需要完整 roofline 视角的融合 kernel）可同时报告两个通道：返回 `OperatorCost::ComputeAndMemory(flops, bandwidthBytes)`。此时 `IsComputeBound()` 与 `IsMemoryBound()` 均为 `true`，`ToString()` 会把两者一并渲染（如 `OperatorCost[flops=N, bandwidthBytes=M]`）。当前尚无此类算子，该工厂为前向预留。

## 5. 系统 / 控制算子

`SyncOp`、`NvtxOp`、`MemoryOp` 这类纯控制 / 内省算子既无计算也无数据搬运，覆写为空代价：

```cpp
// dtorch/core/operators/system/sync_op.h
OperatorCost GetOperatorCost() const override { return OperatorCost{}; }  // HasCost() == false
```

`GetTensorOp` 等确实搬运张量数据的算子**保留基类默认**即可。

## 6. 覆写示例

以 matmul 为例（完整推导见 `matmul_op.cc`）：

```cpp
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
```

## 7. 暂不处理

- **EinsumOp**：当前仅支持单输入、无收缩（transpose/diagonal 类），属纯数据搬运，基类默认的 `bandwidthBytes` 即正确。多输入/收缩的通用 FLOPs 需解析 einsum 文法，留待后续。
- 其余 ~40 个带宽密集算子全部继承基类默认，无需改动。

## 8. 单元测试

`dtorch/tests/test_operator_cost.cc` 覆盖：

- `OperatorCost` 类 API（默认空、`Compute` / `Memory` / `ComputeAndMemory` 工厂、`FromOperands` 字节求和、null-shape 跳过、`DataKind` 字节权重、`ToString` / `operator<<`，含两通道同时设置的混合情形）。
- 各算力密集算子的 FLOPs（matmul 含 batched、linear、outer、sdpa、conv）。
- `SyncOp` 返回空代价。

运行：

```bash
./build/test --gtest_filter=OperatorCostTest.*:OperatorCostOverrideTest.*
```

## 9. 相关源文件

| 文件 | 说明 |
|---|---|
| `dtorch/core/operators/operator_cost.h/.cc` | `OperatorCost` 类（字段、工厂、查询、序列化为字符串） |
| `dtorch/core/operators/operator.h/.cc` | `Operator::GetOperatorCost()` 基类默认实现（委托 `OperatorCost::FromOperands`） |
| `dtorch/core/operators/standard/*.cc` | 算力密集算子的覆写与公式推导注释 |
| `dtorch/tests/test_operator_cost.cc` | 单元测试 |
