# 文档索引

## 快速开始

| 文档 | 内容 |
|---|---|
| [快速开始](../get_started/get_started.md) | 编译安装 DTorch，并验证扩散模型（SD3 / FLUX）的分布式推理 |
| [构建指南](../get_started/how_to_build.md) | 构建与开发环境配置 |
| [测试指南](../get_started/test.md) | Python 单元测试、C++ 单元测试 (gtest)、应用测试（Flux/SD3） |

## 用户指南

| 文档 | 内容 |
|---|---|
| [User Guide](../user_guide/user_guide.md) | 上手 DTorch：先理解 DTensor 核心概念，再学习用 Python API 编写分布式程序 |
| [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md) | DTensor 的 DeviceMesh 与 Placements 概念入门 |
| [Python API 使用指南](../user_guide/python_api_overview.md) | Tensor 创建、算子调用、DTensor 原生支持、Placements 推导、Redistribute、单卡模拟分布式、异步取值 |
| [Module 并行](../user_guide/module_parallel.md) | Linear 的 tp_dim / tp_shard_type、ColumnParallelLinear / RowParallelLinear 子类、Llama DP+TP 完整示例 |
| [扩散模型推理应用](../user_guide/applications.md) | 目录结构、DTorch API 适配修改、ExecuteConfig、分布式、Cache、量化、算子融合 |

## 架构与设计

| 文档 | 内容 |
|---|---|
| [项目概述](project_overview.md) | 分层架构与目录结构总览 |
| [关键概念](key_concept.md) | 三大核心设计理念详解（Single-Client Single-Controller Multi-Worker、DTensor、Eager Graph Architecture） |
| [设计决策](design_decisions.md) | 关键设计决策：为何选择 DTensor + Single-Controller、调度开销的解决、LibTorch 后端等 |
| [Single-Controller 架构](single_controller.md) | Single-Client Single-Controller Multi-Worker 架构详解，与 Multi-Controller 对比 |
| [Distributed Tensor](distributed_tensor.md) | DTensor 的 DeviceMesh 与 Placements 机制，含完整代码示例 |

## Eager Graph 架构

| 文档 | 内容 |
|---|---|
| [架构总览](eager_graph_architecture/eager_graph_architecture.md) | 四层架构（计算图表示 → Kernel 运行时 → GraphExecutor → 集合通讯组件） |
| [Layer 1 · 计算图表示](eager_graph_architecture/logical_graph_representation.md) | Operand / Operator / LogicalGraph |
| [Layer 2 · Kernel 运行时](eager_graph_architecture/kernel_runtime.md) | Blob / Kernel / KernelStream / OperatorAssignInfo |
| [Layer 3 · GraphExecutor](eager_graph_architecture/graph_executor.md) | 单机多线程（GraphConstructor → EagerGraphExecutor → PerDeviceThreadNodeRunner → NaiveRunner） |
| [Layer 3 · 集群模式](eager_graph_architecture/graph_executor_in_cluster.md) | 单机多进程（RemoteRunnerPublisher, RemoteRunnerInProcess, PerDeviceProcessNodeRunner, SubProcess） |
| [Layer 4 · 集合通讯组件](communicate/tensor_communicate.md) | ThreadGroup / TensorStore（Memory/File/Network 三种后端） |
| [异步获取 Tensor 值](eager_graph_architecture/async_get_tensor.md) | GetTensorOp 与 Promise/Future 机制：`to_torch_async()` / TensorFuture |
| [Python Kernel](eager_graph_architecture/python_kernel.md) | 在 C++ Kernel 中调用 Python 代码（GIL 管理、CUDA Stream 保护、类型转换） |
| [序列化](eager_graph_architecture/serialization.md) | Operator 序列化与反序列化：Boost.Serialization 体系、OperatorSerializationPack、跨进程传输 |
| [进程心跳](eager_graph_architecture/process_heart_beat.md) | gRPC 双向心跳、MainProcessHeartBeat / WorkerProcessHeartBeat、进程故障检测与优雅关闭 |
| [ZMQ 通信机制](communicate/zmq.md) | PUB-SUB + PUSH-PULL 双通道、消息协议与顺序保证 |
| [Stream Race Condition](communicate/stream_race_condition.md) | CUDA Stream 同步机制与跨 Stream 的竞态问题 |

## Operator 算子体系

| 文档 | 内容 |
|---|---|
| [算子体系总览](operator/operator.md) | Operator 算子体系总览 |
| [Operator 基类与派生类](operator/operators_class.md) | 基类结构与派生类体系（standard / fused_compile / system） |
| [模板方法模式](operator/operator_template_method_pattern.md) | 算法骨架与所有可重写虚函数详解 |
| [PlacementSignature](operator/placement_signature.md) | 分布式 Placements 映射规则（Builder API、典型实现模式、匹配流程） |
| [算子映射表](operator/operators_mapping.md) | Python↔C++ API↔C++ 核心算子三层映射表 |
| [新增算子开发指南](operator/how_to_add_operator.md) | 注释→核心算子→Kernel→API→测试五步流程 |
| [OperatorCost 算子代价估算](operator/operator_cost.md) | FLOPs / 带宽估算，roofline 分析的基础数据 |

## 调试与优化

| 文档 | 内容 |
|---|---|
| [输出不一致排查](debug_alignment_optimization/debug_output_mismatch.md) | 从粗到细定位 DTorch 与 PyTorch 模型输出不一致的根因 |
| [精度对齐](debug_alignment_optimization/precision_alignment.md) | DTorch 与 PyTorch 输出不一致的已知场景与规避方法 |
| [性能优化](debug_alignment_optimization/performance_optimization.md) | CUDA Kernel Launch 等性能问题的分析与优化 |
