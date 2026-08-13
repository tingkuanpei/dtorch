# LogicalGraph 计算图表示

DTorch 的计算图建立在三个核心抽象之上：**LogicalGraph**（计算图容器）、**Operand**（张量元信息节点）和 **Operator**（计算节点）。Operand 和 Operator 构成一个 DAG（有向无环图）的 LogicalGraph，以拓扑结构表达计算逻辑。

## 1. 架构概览

- **Operand**： 图中的**数据节点**，持有张量的元信息（Shape、Stride、DataKind、DeviceMesh、Placements），不持有实际数据。
- **Operator**： 图中的**计算节点**，持有输入/输出 Operand 的引用，封装单个算子（如 `relu`、`add`、`matmul`）的元信息推导、分布式规则和实际计算逻辑。
- **LogicalGraph**： Operand 和 Operator 构成的 DAG 拓扑容器，仅管理节点引用。

## 2. Operand — 数据节点

**源文件**: `dtorch/core/operand.h`

`Operand` 是计算图中的**数据节点**，表示一个张量的元信息。它不持有实际数据，仅持有张量的描述信息。

### 核心成员

```
Operand
├── 拓扑信息 (Topology)
│   ├── mProducerOp    — 产生此 Operand 的 Operator（有且仅有一个）
│   └── mConsumerOps   — 消费此 Operand 的 Operator 列表（可以有多个）
│
├── 元信息 (Meta Info)
│   ├── mShape         — 张量的形状
│   ├── mStride        — 张量的步长
│   └── mDataKind      — 数据类型（float32, float16, bfloat16, int64 等）
│
├── 分布式信息 (Distribute)
    ├── mDeviceMesh    — N 维设备网格，描述 GPU 拓扑
    └── mPlacementSeq  — 分布式放置序列（Shard/Replicate/Partial）
```

### 拓扑关系

每个 Operand 通过 `mProducerOp` 和 `mConsumerOps` 与 Operator 建立双向连接：


- **ProducerOp**: 指向产生此 Operand 的唯一 Operator。图的输入 Operand 的 ProducerOp 为 `nullptr`。
- **ConsumerOps**: 消费此 Operand 的所有 Operator 列表。图的最终输出 Operand 的 ConsumerOps 为空。

## 3. Operator — 计算节点

**源文件**: `dtorch/core/operators/operator.h`

`Operator` 是计算图中的**计算节点**。用户调用的每个算子（如 `relu`、`add`、`matmul`）在图中都对应一个 Operator 节点，它连接输入 Operand 与输出 Operand：

```
mInputOperands[0] ──┐
mInputOperands[1] ──┤
        ...         ├──► Operator ──► mOutputOperands[0]
mInputOperands[N] ──┘                  mOutputOperands[...]
```

`Operator` 是基类，每个算子都有对应的派生类（如 `ConvOp`、`LinearOp`、`ReduceOp`），使用模板方法模式，基于基类实现各自的行为。

每个 Operator 派生类封装了：

- **元信息推导** — 根据输入 Operand 的元信息（Shape、DeviceMesh、Placements 等）推导输出 Operand 的元信息，此过程不涉及实际数据
- **分布式规则** — 声明该算子支持的输入/输出 Placements 组合（PlacementSignature）

每个 Operator 还会持有 **`OpParam`**，由于描述该算子所需的全部参数——如卷积的 `ConvParam` 携带 `kernelSize`、`pads`、`strides` 等。

## 4. LogicalGraph — DAG 拓扑容器

**源文件**: `dtorch/core/graph/logical_graph.h`

`LogicalGraph` 是计算图的顶层容器，使用两个 `unordered_map` 管理图中的所有节点：

| 成员 | 类型 | 说明 |
|---|---|---|
| `mOperatorMap` | `unordered_map<const Operator*, shared_ptr<Operator>>` | 算子映射，key 为原始指针，value 为智能指针，支持 O(1) 查找 |
| `mOperandMap` | `unordered_map<const Operand*, shared_ptr<Operand>>` | 操作数映射，同上 |

### 关键操作

| 方法 | 说明 |
|---|---|
| `AddOperator(op)` | 向图中添加一个算子节点 |
| `AddOperand(operand)` | 向图中添加一个操作数节点 |
| `DeleteOperator(op)` | 删除算子节点（仅移除引用，不更新图拓扑） |
| `DeleteOperand(operand)` | 删除操作数节点（仅移除引用，不更新图拓扑） |

`LogicalGraph` 本身**不维护图的拓扑遍历顺序**，拓扑排序由 `GraphTraversalSequence` 完成。

### 生命周期

在 Eager 模式下，`LogicalGraph` 由 `EagerGraphExecutor` 持有（见 `eager_graph_executor.h:97`）。图会**动态增长**：每当用户调用 `api::cpp::functional` 接口时添加新节点；每个新增节点只执行一次，执行完毕后被销毁。

## 5. 图构建

### GraphConstructor

**源文件**: `dtorch/core/graph/graph_constructor.h`

`GraphConstructor` 是 Python API 与核心引擎之间的**桥梁**。当用户调用 `dtorch.functional.relu()` 等 API 时：

1. Python 层通过 nanobind 调用 C++ API
2. API 层创建 `OpParam` 并调用 `GraphConstructor::AddOperator()`
3. `GraphConstructor` 实例化对应的 `Operator`，执行 `Infer()` 推导元信息
4. 通过 `SendOperatorToExecutor()` 将算子发送给 `EagerGraphExecutor`

### GraphTraversalSequence

**源文件**: `dtorch/core/graph/graph_traversal_sequence.h`

`GraphConstructor` 创建 `Operators` 的顺序会保存在 `class GraphTraversalSequence`。`class GraphTraversalSequence` 封装了基于 `std::list` 的算子遍历序列，提供：
- O(1) 查找、插入、删除（通过 `mNodeMap` 辅助映射）
- 双向迭代器接口
- 执行顺序的维护

## 6. 源文件索引

| 文件 | 说明 |
|---|---|
| `dtorch/core/graph/logical_graph.h` `.cc` | `LogicalGraph` 类 — DAG 拓扑容器 |
| `dtorch/core/operand.h` `.cc` | `Operand` 类 — 张量元信息数据节点 |
| `dtorch/core/operators/operator.h` `.cc` | `Operator` 基类 — 计算节点 |
| `dtorch/core/graph/graph_constructor.h` | `GraphConstructor` 类 — 图构建器 |
| `dtorch/core/graph/graph_traversal_sequence.h` | `GraphTraversalSequence` 类 — 拓扑遍历序列 |
