# Operator 模板方法模式

`class Operator` 采用**模板方法模式（Template Method Pattern）**，在基类中定义了算子推断和执行的算法骨架，派生类通过重写特定的虚函数来实现各自的功能。

## 1. 算法骨架

`Operator` 基类通过 `Infer()` 和 `NewOperatorOrThrow()` 定义了算子的标准生命周期：

```
Infer()
  └── InferOutput()
        ├── Assert(GetOutputSize() == InferOutputSize())  // 步骤1: 校验输出 Operand 数量
        ├── CheckInput()                        // 步骤2: 检查输入合法性
        │     ├── CheckInputSameDataKind()      //   - 输入 DataKind 一致性
        │     ├── CheckInputAllDistributedOrNot()//   - 分布式/非分布式一致性
        │     ├── CheckInputSameDeviceMesh()    //   - DeviceMesh 一致性
        │     └── CheckInputDistributedSpec()   //   - Placements 合法性校验
        ├── InferOutputMetaInfo()               // 步骤3: ★ 必须重写 — 推断输出 Shape/Stride/DataKind
        └── if (!SkipDistributedSpec...())      // 步骤4: 分布式推断（默认对分布式输入执行）
              └── InferOutputDistributedSpecFromPlacementSignature()
                    ├── GetPlacementSignature()  //   - 获取 Placement 映射规则
                    ├── Match(inputs, outputs)   //   - 逐维度匹配
                    └── SetDeviceMeshAndPlacementSeq() // - 设置输出分布式信息
```

## 2. 重写基类函数指南

派生类根据自身需要，按需重写以下虚函数。下面按重写频率从高到低排列。

### 2.1 InferOutputMetaInfo() — 必须重写 ⭐

```cpp
virtual void InferOutputMetaInfo() const = 0;  // 纯虚函数
```

这是**必须重写的纯虚函数**。需要根据输入 Operand 的 Shape、Stride、DataKind 和算子参数，推断并设置输出 Operand 的元信息。

**典型实现示例** — 激活函数（输出与输入同 Shape、同 Stride、同 DataKind）：

```cpp
// 来自 dtorch/core/operators/standard/activation_op.cc
void ActivationOp::InferOutputMetaInfo() const {
    OperandY()->SetShapeAndStrideV2(OperandX()->GetShape(), OperandX()->GetStride());
    OperandY()->SetDataKind(OperandX()->GetDataKind());
}
```

**卷积算子** — 需要计算输出 Shape（基于 input/kernel/dilation/stride/pad）：

```cpp
// 来自 dtorch/core/operators/standard/conv_op.cc
void ConvOp::InferOutputMetaInfo() const {
    const auto& param = GetOpParam<ConvParam>();
    // ... 计算输出 Shape、Stride
    OperandY()->SetShapeAndStrideV2(outputShape, outputStride);
    OperandY()->SetDataKind(OperandX()->GetDataKind());
}
```

**关键 API**：

| API | 用途 |
|---|---|
| `SetShapeAndStride(shape, stride)` | 同时设置 Shape 和 Stride |
| `SetDataKind(dataKind)` | 设置数据类型 |
| `GetShape()`, `GetStride()`, `GetDataKind()` | 获取输入 Operand 的元信息 |
| `GetOpParam<DerivedOpParam>()` | 获取类型安全的算子参数引用 |

### 2.2 GetDescribeString() — 算子描述字符串 ⭐

```cpp
virtual std::string GetDescribeString() const { return OpTypeToString(GetOpType()); }
```

**每个派生类都应重写此方法**，用于调试输出、日志记录和 Graph 可视化。基类默认实现仅返回算子类型名称（如 `"Matmul"`），缺少上下文信息，不利于问题排查。

> 详细的实现指南和代码示例见 [How To Add Operator §2.4](how_to_add_operator.md#24-实现-operator-类并重写虚函数)。

**重写要求**：

- 至少包含：算子名 + 输入 Shape + 输出 Shape
- 对于有参数的算子，附带关键参数信息（如激活类型、卷积参数等）
- 使用 `std::stringstream` 构建，避免直接字符串拼接

**典型实现模式**：

| 算子类型 | 包含信息 | 示例输出 |
|---|---|---|
| 无参数算子 | 算子名 + 输入 Shape → 输出 Shape | `"Matmul([4,8] × [8,6] → [4,6])"` |
| 有参数算子（Element-wise） | 算子名 + 参数类型/值 | `"Activation: ReLU(inplace=false)"` |
| 有参数算子（Reduce） | 算子名 + reduce dims + keepdim | `"Reduce: Sum(dim=[0,1], keepdim=true)"` |
| 创建算子 | 算子名 + 输出 Shape | `"Create([2, 3, 224, 224], float32)"` |

**完整示例**：

```cpp
// 无参数算子：展示输入→输出 Shape
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

// 有参数算子：展示参数信息
std::string BroadcastBinaryOp::GetDescribeString() const {
    const auto& param = GetOpParam<BroadcastBinaryParam>();
    std::stringstream ss;
    ss << GetOpType() << ": " << BroadcastBinaryKindToString(param.binaryKind);
    return ss.str();
}
```

### 2.3 InferOutputSize() — 可选重写

```cpp
virtual size_t InferOutputSize() const { return 1; }
```

默认返回 `1`，表示输出一个 Operand。对于产生多个输出或零输出的算子需要重写：

```cpp
// NvtxOp 不产生任何输出 Tensor
size_t NvtxOp::InferOutputSize() const override { return 0; }

// ChunkOp 产生多个输出 Tensor
size_t ChunkOp::InferOutputSize() const override { /* 根据 chunks 参数计算 */ }
```

### 2.4 SkipDistributedSpecFromPlacementSignature() — 跳过分布式推断

```cpp
virtual bool SkipDistributedSpecFromPlacementSignature() const;
```

**默认实现**：当第一个输入 Operand 是非分布式的（`!IsDistributed()`）返回 `true`，即跳过 Placement 推断。

**何时重写为返回 `true`**：
- 没有输入 Tensor 的算子（如 `CreateOp`），基类的默认实现会 assert 失败
- Element-wise 算子（如 `ActivationOp`），输出 Placement 与输入完全相同，不需要 Placements 匹配
- 系统算子（如 `NvtxOp`），不涉及 Tensor 分布式逻辑

```cpp
// CreateOp: 没有输入 Tensor
// ActivationOp: Element-wise，输出 Placement 直接从输入拷贝即可
// NvtxOp: 系统算子
bool SkipDistributedSpecFromPlacementSignature() const override { return true; }
```

### 2.5 GetPlacementSignature() — 定义分布式规则

```cpp
virtual PlacementSignature GetPlacementSignature() const;
```

当算子输入是 DTensor 且不能简单复制输入 Placement 到输出时，需要重写此函数。

> **详细说明、Builder API 和完整实现示例见 [PlacementSignature 文档](placement_signature.md)**。

典型场景对比：

| 算子类型 | 是否重写 | 示例 |
|---|---|---|
| Element-wise (激活、Dropout 等) | 不重写，通过 `SkipDistributedSpec` 跳过 | `ActivationOp` |
| Shape-preserving (Softmax, Normalization) | 重写 | `SoftmaxOp` — 沿非 softmax 维度传递 Shard |
| Reduce (sum, mean) | 重写 | `ReduceOp` — 被 reduce 维度 Shard→Replicate |
| Broadcast Binary (add, mul) | 重写 | `BroadcastBinaryOp` — 处理广播维度对齐 |
| MatMul / Linear | 重写 | `MatmulOp` — 引入 Partial 表示规约部分和 |
| 创建算子 (zeros, ones) | 不重写 | `CreateOp` — 无输入，跳过分布式推断 |

### 2.6 Compute() — Kernel 计算

```cpp
virtual void Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const;
```

绝大部分算子均需重写此函数，少部分算子因为需要使用 ThreadGroup 或指定 cuda stream，需自定义 Kernel。可参考 `docs/how_to_add_operator.md`

```cpp
// ActivationOp: 调用 LibTorch 的 activation 函数
void ActivationOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    const auto& param = GetOpParam<ActivationParam>();
    // 根据 param.activationType 调用对应的 LibTorch 函数
    switch (param.activationType) {
        case ActivationType::kReLU:
            outputs[0] = torch::relu(inputs[0].value());
            break;
        // ...
    }
}
```

**不重写 Compute() 的例外**：
- `CopyOp、ConvertOp、ViewOp` — 由其他自定义的 Kernel 处理

### 2.7 IsRequireInputSameDataKind() — 输入 DataKind 检查

```cpp
virtual bool IsRequireInputSameDataKind() const { return true; }
```

默认要求所有输入 Tensor 的 DataKind 一致。对于混合数据类型的算子（如 `WhereOp` 的 condition 可以是 `bool`），重写返回 `false`：

```cpp
// WhereOp 的 condition 输入是 bool 类型，与 X/Y 输入的 DataKind 不同
bool IsRequireInputSameDataKind() const override { return false; }
```

### 2.8 InferOperatorAssignInfo() — 自定义 Kernel 分配

```cpp
virtual void InferOperatorAssignInfo();
```

默认实现根据输入/输出 Operand 的 DeviceMesh 自动生成 `KernelStreamKey`。对于没有输入输出的算子（如 `NvtxOp`），需要重写：

```cpp
// NvtxOp: 没有输入输出 Operand，需要手动指定执行 Device
void NvtxOp::InferOperatorAssignInfo() override {
    const auto& param = GetOpParam<NvtxParam>();
    for (auto deviceId : param.deviceMesh.GetDeviceIdSet()) {
        KernelStreamKey streamKey;
        streamKey.Init(DeviceKind::kGpu, deviceId, KernelStreamType::kCompute);
        mOperatorAssignInfo.Insert(streamKey);
    }
}
```

### 2.9 CheckInput() — 额外的输入校验

```cpp
virtual void CheckInput() const;
```

基类已执行 `CheckInputSameDataKind`、`CheckInputAllDistributedOrNot`、`CheckInputSameDeviceMesh`、`CheckInputDistributedSpec` 等通用检查。派生类可重写以添加额外的校验逻辑。

### 2.10 UpdateIOOperandTopology() — 更新拓扑关系

```cpp
virtual void UpdateIOOperandTopology();
```

默认实现将输入 Operand 的消费关系（`AddConsumesOp`）和输出 Operand 的生产关系（`SetProducerOp`）建立起来。一般情况下不需要重写。

### 2.11 GetOperatorCost() — 代价估算

```cpp
virtual OperatorCost GetOperatorCost() const;
```

估算算子在**当前输入/输出 Shape** 下的算力需求（FLOPs）或带宽需求（bytes），供后续算力/带宽利用率（roofline）分析使用。返回值 `OperatorCost` 有两条互斥通道：算力密集算子填 `flops`，带宽密集算子填 `bandwidthBytes`，通常只填一条。详见 [算子代价估算 (OperatorCost)](operator_cost.md)。

**前置条件（重要）**：`GetOperatorCost()` 在 `Infer()` **之后**调用，此时**所有输入与输出 Operand 的 Shape 都已推断完毕、可直接读取**。因此覆写实现应当直接读取 Operand 的 Shape（例如 `OperandY()->GetShape()`），而**不要重新推导输出 Shape**。

**基类默认实现**为带宽密集（memory-bound）：`bandwidthBytes` = 所有非 null 输入 + 输出 Operand 的字节之和（通过 `OperatorCost::FromOperands(...)` 计算）。这意味着 `add`、`copy`、`reshape`、激活、归一化等绝大多数 element-wise / 数据搬运算子**无需覆写**即自动正确。

**何时覆写**：

| 算子类型 | 是否覆写 | 返回值 |
|---|---|---|
| 算力密集（matmul、conv、sdpa、linear、outer 等） | ★ 覆写 | `OperatorCost::Compute(flops)` |
| 带宽密集（add、activation、norm、reduce、reshape、copy 等） | 不覆写 | 基类默认（`bandwidthBytes`） |
| 纯控制 / 内省（sync、nvtx、memory 等） | 覆写为空 | `OperatorCost{}` |

**覆写示例** — 矩阵乘（按 MAC 约定，每次乘累加计 2 FLOPs）：

```cpp
// 来自 dtorch/core/operators/standard/matmul_op.cc
OperatorCost MatmulOp::GetOperatorCost() const {
    DDebugAssert(GetInputSize() == 2);
    const Shape& shapeA = OperandA()->GetShape();
    const Shape& shapeB = OperandB()->GetShape();

    // 拆分广播维与计算维（复用私有 SplitBroadcastShape）
    auto [broadcastShapeA, computeShapeA] = SplitBroadcastShape(shapeA);
    auto [broadcastShapeB, computeShapeB] = SplitBroadcastShape(shapeB);

    int64_t batch = static_cast<int64_t>(Shape::BroadcastOutputShape(broadcastShapeA, broadcastShapeB).Count());
    int64_t M = (computeShapeA.NumAxis() == 2) ? static_cast<int64_t>(computeShapeA[0]) : 1;
    int64_t K = static_cast<int64_t>(computeShapeA[computeShapeA.NumAxis() - 1]);
    int64_t N = (computeShapeB.NumAxis() == 2) ? static_cast<int64_t>(computeShapeB[1]) : 1;

    return OperatorCost::Compute(2 * batch * M * N * K);
}
```

**控制算子覆写为空**：

```cpp
// 来自 dtorch/core/operators/system/sync_op.h
OperatorCost GetOperatorCost() const override { return OperatorCost{}; }
```

> 各算力密集算子的完整 FLOPs 公式推导（含 MAC 约定、SdpaOp 的 GQA 处理、ConvOp 读取已推断输出 Shape 等）见 [算子代价估算 (OperatorCost) §4](operator_cost.md#4-算力密集算子覆写)。
