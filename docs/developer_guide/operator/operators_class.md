# Operator 类

`class Operator` 是 DTorch 中所有算子的抽象基类，定义在 `dtorch/core/operators/operator.h`。它采用**模板方法模式（Template Method Pattern）**，在基类中定义了算子推断和执行的算法骨架，派生类通过重写特定的虚函数来实现各自的功能。

## 1. Operator 基类

### 1.1 核心成员

```cpp
class Operator {
protected:
    std::string mOpName;                          // 算子名称
    uint64_t mUniqueId;                           // 全局唯一 ID
    OperatorType mOpType;                         // 算子类型枚举
    std::shared_ptr<OpParam> mOpParam;            // 算子参数（多态）

    OperandArray mInputOperands;                  // 输入 Operand 列表
    OperandArray mOutputOperands;                 // 输出 Operand 列表

    OperatorAssignInfo mOperatorAssignInfo;       // Kernel 分配信息（Stream、Device 等）
};
```

关键依赖类型：

| 类型 | 位置 | 说明 |
|---|---|---|
| `OperatorType` | `operator_param.h` | 枚举所有算子类型（`kActivation`, `kConv`, `kReduce`, …），通过 `DTORCH_FOREACH_OPERATOR_TYPE` 宏统一定义 |
| `OpParam` | `operator_param.h` | 算子参数的抽象基类，通过 `OperatorType` 枚举标识算子类型。所有派生类参数结构体都继承自 `OpParam` |
| `Operand` | `dtorch/core/operand.h` | 计算图中的张量节点，持有 Shape、Stride、DataKind、DeviceMesh、PlacementSeq 等元信息 |
| `OperatorAssignInfo` | `operator_assign_info.h` | 描述算子在哪些 Stream（Device + StreamType）上执行 |


## 2. 派生类体系

所有算子分为三大类别，分别位于对应的子目录中：

```
dtorch/core/operators/
├── operator.h / .cc              ← Operator 基类
├── operator_param.h / .cc        ← OpParam 基类 + OperatorType 枚举
├── operator_factory.h / .cc      ← OperatorFactory 工厂类
├── placement_signature.h / .cc   ← PlacementSignature 匹配引擎
│
├── standard/                     ← 标准算子（LibTorch 后端实现）
│   ├── activation_op.h           ReLU, GELU, SiLU 等激活函数
│   ├── broadcast_binary_op.h     add, mul, sub, div 等广播二元运算
|   ...
│
├── fused_compile/                ← 融合编译算子（性能优化）
│   ├── apply_rotary_emb_op.h     RoPE 位置编码融合
│   ├── silu_linear_chunk.h       SiLU + Linear 融合
│   └── layer_norm_mul_add.h      LayerNorm + Mul + Add 融合
│
└── system/                       ← 系统算子（不产生输出 Tensor）
    ├── nvtx_op.h                 NVTX 性能标记
    └── memory_op.h               显存管理
```

此外还有两个特殊算子位于顶层：

- **`FuseOp`** (`fuse_op.h`) — 表示一个子图的融合节点，由 `LogicalGraph` 和主导算子类型组成
- **`SubGraphOp`** (`subgraph_op.h`) — 封装子图，内部持有 `LogicalGraph`

## 3 算法骨架：模板方法模式

> **详细说明、算法骨架和所有可重写虚函数的完整指南见 [Operator 模板方法模式文档](operator_template_method_pattern.md)。**

## 4. PlacementSignature 分布式规则

PlacementSignature 是 DTorch 分布式算子的核心机制，用于描述算子输入和输出 Tensor 的 Placement 映射关系。

> **PlacementSignature 的完整说明（包括 Builder API、典型实现模式、匹配流程）请参见 [placement_signature.md](placement_signature.md)。**

## 5. Operator 生命周期

一个 Operator 从创建到执行的完整流程：

```
1. 工厂构造
   OperatorFactory::NewOperatorOrThrow(opParamPtr, inputOperands)
     ├── ConstructOperator(opParamPtr)     ← 根据 OpType 创建对应的派生类实例
     ├── SetUniqueId() / AssignUniqueId()
     ├── SetInputOperands(inputOperands)
     ├── CreateOutputOperands()            ← 根据 InferOutputSize() 创建空 Output Operand
     │
2. 推断阶段
     ├── Infer()
     │     ├── InferOutput()
     │     │     ├── CheckInput()           ← 校验输入合法性
     │     │     ├── InferOutputMetaInfo()  ← ★ 派生类实现：推断 Shape/Stride/DataKind
     │     │     └── InferOutputDistributedSpecFromPlacementSignature()  ← 分布式信息推断
     │     └── InferOperatorAssignInfo()    ← 分配 Kernel→Stream 映射
     │
3. 拓扑注册
     └── UpdateIOOperandTopology()          ← 建立 Operand 的生产者/消费者关系
```

在构图阶段（步骤 1-3）完毕后，算子被添加到计算图中。执行阶段由 `EagerGraphExecutor` 驱动，Worker 线程调用 `Compute()` 执行实际计算：

```
4. 执行阶段（运行在 Worker 线程上）
   Operator::Compute(inputs, outputs)       ← 派生类实现：调用 LibTorch 执行 CUDA Kernel
```

## 6. 添加新 Operator 的步骤

> **详细步骤请参见 [how_to_add_operator.md](how_to_add_operator.md)**。

## 7. 文件索引

| 文件 | 说明 |
|---|---|
| `dtorch/core/operators/operator.h` | `Operator` 基类声明 |
| `dtorch/core/operators/operator.cc` | `Operator` 基类实现（推断、校验、拓扑） |
| `dtorch/core/operators/operator_param.h` | `OpParam` 基类 + `OperatorType` 枚举 |
| `dtorch/core/operators/operator_factory.h/.cc` | `OperatorFactory` 单例工厂 + 算子注册 |
| `dtorch/core/operators/placement_signature.h/.cc` | `PlacementSignature` + `Builder` 实现 |
| `dtorch/core/operators/operator_assign_info.h` | `OperatorAssignInfo`（Stream 分配） |
| `dtorch/core/operators/subgraph_op.h` | `SubGraphOp` 子图封装 |
| `dtorch/core/operators/fuse_op.h` | `FuseOp` 融合算子 |
| `dtorch/core/operators/standard/` | 标准算子实现目录 |
| `dtorch/core/operators/fused_compile/` | 融合编译算子目录 |
| `dtorch/core/operators/system/` | 系统算子目录 |
