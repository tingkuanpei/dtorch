# Operator（算子）

Operator 是 DTorch 计算图中的**计算节点**，封装了单个操作（如 `relu`、`add`、`matmul`、`conv2d` 等）的元信息推断、分布式规则和实际计算逻辑。

**理解 Operator 的核心入口是 [operators_class.md](operators_class.md)**，它涵盖了 Operator 基类结构、派生类体系（standard / fused_compile / system）以及完整生命周期。建议先阅读该文档建立整体认知，再按需查阅以下专题文档。

## 1. 文档索引

| 文档 | 内容 | 适用场景 |
|---|---|---|
| [operator_template_method_pattern.md](operator_template_method_pattern.md) | 模板方法模式的算法骨架、所有可重写虚函数的详细说明 | 理解 `Infer()` 流程，或重写特定虚函数 |
| [placement_signature.md](placement_signature.md) | 分布式 Placements 映射规则（Builder API、典型实现模式、匹配流程） | 实现分布式算子的 `GetPlacementSignature()` |
| [operators_mapping.md](operators_mapping.md) | Python 接口 ↔ C++ API 接口 ↔ C++ 核心算子的完整映射表 | 查找某个 Python 接口对应的底层实现 |
| [how_to_add_operator.md](how_to_add_operator.md) | 新增算子的完整步骤（4 层改造 + 完整示例） | 开发新算子 |
| [operator_cost.md](operator_cost.md) | 算子代价估算（FLOPs / 带宽）、`OperatorCost` 类、覆写约定与公式推导 | 性能分析、实现算力密集算子的 `GetOperatorCost()` |

## 2. 核心源文件

| 文件 | 说明 |
|---|---|
| `dtorch/core/operators/operator.h` | `Operator` 基类声明（模板方法骨架、虚函数接口） |
| `dtorch/core/operators/operator.cc` | `Operator` 基类实现（推断、校验、拓扑注册） |
| `dtorch/core/operators/operator_cost.h/.cc` | `OperatorCost` 类（FLOPs / 带宽代价估算） |
| `dtorch/core/operators/operator_param.h` | `OpParam` 基类 + `OperatorType` 枚举（通过 `DTORCH_FOREACH_OPERATOR_TYPE` 宏统一定义） |
| `dtorch/core/operators/operator_factory.h/.cc` | `OperatorFactory` 单例工厂 + 算子注册 |
| `dtorch/core/operators/placement_signature.h/.cc` | `PlacementSignature` + `Builder` 实现 |
| `dtorch/core/operators/operator_serialization_pack.h/.cc` | 算子序列化（支持跨进程传输） |
| `dtorch/core/operators/operator_assign_info.h` | `OperatorAssignInfo`（Kernel → Stream 分配） |

## 3. 相关文档

- [Distributed Tensor](../distributed_tensor.md) — DTensor 的 DeviceMesh 与 Placements 机制
