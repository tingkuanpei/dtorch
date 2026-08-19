# PlacementSignature

PlacementSignature 是 DTorch 分布式算子的核心机制，描述了算子**输入 Tensor 的 Placement**与**输出 Tensor 的 Placement**之间的映射关系，因此可以根据输入 Tensor 的 Placements 推断出输出 Tensor 的 Placements。

## 1. 背景

DTorch 原生支持 DTensor（Distributed Tensor）。每个 DTensor 通过 `DeviceMesh` 和 `PlacementSeq` 描述其在分布式集群上的分布方式。`PlacementSeq` 是一个维度序列，每个维度可以是：

| Placement 类型 | 含义 | 构造函数 |
|---|---|---|
| `Replicate()` | 该维度在所有设备上完整复制 | `"R"` |
| `Shard(dim)` | 该维度沿第 `dim` 轴切分到不同设备上 | `"S{index}"`，如 `"S0"`, `"S1"` |
| `Partial()` | 该维度在各设备上是部分结果（如矩阵乘法的部分和），需要 reduce 后才能得到完整值 | `"P"` |
| `Optional()` | 通配符，仅在 PlacementSignature 内部使用, 用于算子的可选输入（如 matmul 的 bias 输入） | — |

一个二维 Tensor 的 `PlacementSeq` 例如 `["S0", "R"]` 表示：第 0 维 Shard 到各设备上，第 1 维在各设备上完整复制。

## 2. 核心概念

`PlacementSignature` 描述了一个算子**输入 Tensor 的 Placement**与**输出 Tensor 的 Placement**之间的映射关系。它是多条 (InputPlacementSignature → OutputPlacementSignature) 规则组成的集合。

在算子执行 `InferOutput()` 流程时，框架会：

1. 检查是否需要推断分布式信息（`SkipDistributedSpecFromPlacementSignature()`）
2. 如果需要则执行推断函数（ `InferOutputDistributedSpecFromPlacementSignature()`）
    - 调用 `GetPlacementSignature()` 获取该算子的 PlacementSignature
    - 调用 `PlacementSignature::Match()` 将输入 Tensor 的实际 `PlacementSeq` 与签名规则进行匹配
    - 匹配成功后，自动设置输出 Tensor 的 `PlacementSeq`。
    - 自动设置输出 Tensor 的 `DeviceMesh`。（Operator 所有输入输出 Tensor 的 DeviceMesh 必须一致，因此直接拷贝输入的 DeviceMesh 即可。）

**匹配是按维度独立进行的**：对于 Shape 的每一维，提取所有输入 Tensor 在该维的 Placement，组合成 hash key，在签名表中查找对应的输出 Placement。

## 3. 何时需要实现

当算子**输入 Tensor 是分布式（DTensor）的**，且输出的 `PlacementSeq` 不能简单地从输入复制时，需要实现 `GetPlacementSignature()`。如果算子**不需要设置**分布式信息，选择以下两种方式之一：

- **跳过整个流程**：重写 `SkipDistributedSpecFromPlacementSignature()` 返回 `true`。适用于：
  - 非分布式 Tensor 上的操作（默认行为）
  - Element-wise 算子（如 `ActivationOp`），输出 Placement 与输入完全相同，不需要额外规则
  - 没有输入 Tensor（如 `CreateOp`）。

  ```cpp
  // 来自 dtorch/core/operators/standard/activation_op.h
  bool SkipDistributedSpecFromPlacementSignature() const override { return true; }
  ```

- **使用默认签名**：不重写任何函数，使用 `Operator` 基类的默认实现（空的 PlacementSignature），仅适用于输入输出 Placement 都是 `Replicate` 的简单场景。

## 4. Builder 模式

`PlacementSignature` 通过 `Builder` 模式构建，API 如下：

```cpp
PlacementSignature::Builder builder(inputSize, outputSize);

// 添加一条规则：一组输入 Placement → 一组输出 Placement
builder.AddInput(placement)            // 必选输入 placement
    .AddOptionalInput(placement)       // 可选输入 placement（匹配时可忽略），如 matmul 的 bias 输入
    .AddOutput(placement)              // 对应的输出 placement
    .Build();                          // 完成当前规则，可继续添加下一条

// 所有规则添加完毕后
return builder.Finish();               // 自动补全 Replicate 规则并生成 PlacementSignature
```

**重要**：必须为每种合法的输入 Placement 组合添加规则。`Builder::Finish()` 会自动将未被显式规则的输入补全为 `Replicate` → `Replicate` 规则，确保签名是完备的（即对所有输入组合都有明确的输出）。**建议显式声明所有规则**。

**`AddOptionalInput`** 用于某些输入是可选的场景（如 bias），匹配时会尝试忽略 Optional 输入和包含 Optional 输入两种情况。

## 5. 典型实现模式

### 模式 1：沿 Shard 维度传递（Shape-preserving 算子）

对于不改变 Shape 维度顺序的算子（如 `SoftmaxOp`），Shard 维度直接从输入传到输出，但 softmax dim 除外（Softmax 需要在完整维度上计算）：

```cpp
// 来自 dtorch/core/operators/standard/softmax_op.cc
PlacementSignature SoftmaxOp::GetPlacementSignature() const {
    const auto& param = GetOpParam<SoftmaxParam>();
    const Shape& inputShape = OperandX()->GetShape();
    size_t dim = Operator::GetValidDim(inputShape, param.dim);

    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    for (size_t i = 0; i <= inputShape.NumAxis(); i++) {
        if (dim == i) {
            continue;  // softmax dim: 不添加 Shard 规则，由 Finish() 补全为 R→R
        }
        builder.AddInput(Shard(i)).AddOutput(Shard(i)).Build();
    }
    return builder.Finish();
}
```

### 模式 2：Reduce 类算子（Shard → Replicate）

Reduce 类算子（sum, mean 等）对被 reduce 的维度执行 Shard→Replicate，对其他维度执行 Shard→Shard（传递）：

```cpp
// 来自 dtorch/core/operators/standard/reduce_op.cc
PlacementSignature ReduceOp::GetPlacementSignature() const {
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    const Shape& inShape = OperandX()->GetShape();
    const auto& param = GetOpParam<ReduceParam>();

    if (param.dim.size() == 0) {  // reduce all dims
        for (size_t i = 0; i < inShape.NumAxis(); i++) {
            builder.AddInput(Shard(i)).AddOutput(Replicate()).Build();
        }
    } else {
        std::unordered_set<size_t> dimSet = param.GetDimSet(inShape);
        for (size_t inIdx = 0, outIdx = 0; inIdx < inShape.NumAxis(); inIdx++) {
            if (dimSet.find(inIdx) != dimSet.end()) {
                builder.AddInput(Shard(inIdx)).AddOutput(Replicate()).Build();  // reduced dim
            } else {
                builder.AddInput(Shard(inIdx)).AddOutput(Shard(outIdx)).Build(); // preserved dim
            }
            if (param.keepdim || dimSet.find(inIdx) == dimSet.end()) outIdx++;
        }
    }
    return builder.Finish();
}
```

### 模式 3：Broadcast Binary 类算子（处理广播维度）

二元算子（add, mul 等）需要处理 Tensor 广播，涉及维度补齐和 Shard 传播：

```cpp
// 来自 dtorch/core/operators/standard/broadcast_binary_op.cc
void BroadcastBinaryOp::AddBroadcastBinaryOpPlacementSignature(
    PlacementSignature::Builder& builder, const Shape& shapeA, const Shape& shapeB) {
    // 广播维度对齐后：
    // 1. A 独有维度: Shard(A) + R → Shard(Y)
    // 2. B 独有维度: R + Shard(B) → Shard(Y)
    // 3. 共享维度: Shard(A) + Shard(B) → Shard(Y)
    // ...
}

PlacementSignature BroadcastBinaryOp::GetPlacementSignature() const {
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    AddBroadcastBinaryOpPlacementSignature(builder, OperandA()->GetShape(), OperandB()->GetShape());
    return builder.Finish();
}
```

### 模式 4：MatMul/Linear 类算子（Shard + Partial 引入通信）

矩阵乘法涉及 Shard 维度的组合规约，需要引入 `Partial` 来表示部分和结果：

```
// GEMM PlacementSignature 参考表：
// |  X  |  W  |  Y  |
// | S0  |  R  | S0  |  ← batch dim Shard 传递
// |  R  | S1  | S1  |  ← hidden dim Shard 传递
// | S1  | S0  |  P  |  ← 规约维度组合产生 Partial（需要后续 all-reduce）
// |  P  |  R  |  P  |
// |  R  |  P  |  P  |
```

```cpp
// 来自 dtorch/core/operators/standard/matmul_op.cc（简化）
PlacementSignature MatmulOp::GetPlacementSignature() const {
    PlacementSignature::Builder builder(GetInputSize(), GetOutputSize());
    // ... broadcast shape 部分复用 BroadcastBinaryOp 逻辑
    // 2D GEMM 规则：
    builder.AddInput(Shard(M-2)).AddInput("R").AddOutput(Shard(M-2)).Build();
    builder.AddInput("R").AddInput(Shard(M-1)).AddOutput(Shard(M-1)).Build();
    builder.AddInput(Shard(M-1)).AddInput(Shard(M-2)).AddOutput("P").Build();  // 产生 Partial
    return builder.Finish();
}
```

更多完整示例可参考：

- `dtorch/core/operators/standard/concat_op.cc` — 多输入多维度 Shard 传递（含 SubSplitCoordinates）
- `dtorch/core/operators/standard/linear_op.cc` — 三元输入 + OptionalInput 用法
- `dtorch/core/operators/standard/embedding_op.cc` — 查表操作的 Shard 映射
- `dtorch/core/operators/standard/scaled_dot_product_attention_op.cc` — 多头注意力的 batch/head Shard

## 6. 核心流程总结

```
InferOutput()
  ├── InferOutputMetaInfo()           // 子类实现：计算 Shape、Stride、DataKind
  └── if (!SkipDistributedSpec...)    // 子类可选跳过
      └── InferOutputDistributedSpecFromPlacementSignature()
          ├── GetPlacementSignature() // 子类实现：返回 PlacementSignature
          ├── Match(inputs, outputs)  // 逐维匹配输入→输出 Placement
          └── SetDeviceMeshAndPlacementSeq()  // 设置输出 Operand 的分布式信息
```

## 7. 不匹配时的错误处理

当输入 Tensor 的实际 `PlacementSeq` 无法匹配算子定义的 PlacementSignature 时，框架会抛出异常，包含：

- 算子类型
- 各输入 Tensor 的实际 Placement
- 该算子支持的 PlacementSignature 完整列表

此时用户需要调整代码（如调用 `tensor.redistribute()` 来改变 Tensor 的分布方式），某些算子（如 `scaled_dot_product_attention`）在文档中会单独说明其自动插入 redistribute 的行为。

## 8. 实现位置

`PlacementSignature` 的核心实现位于：

- `dtorch/core/operators/placement_signature.h` — `PlacementSignature` 类及 `Builder` 定义
- `dtorch/core/operators/placement_signature.cc` — 匹配逻辑实现
- `dtorch/api/cpp/distributed_spec.h` — `Placement`、`PlacementSeq` 类型定义
- `dtorch/core/operators/operator.h` — `Operator` 基类中的接口声明
- `dtorch/core/operators/standard/*.cc` — 各算子的 `GetPlacementSignature()` 实现
