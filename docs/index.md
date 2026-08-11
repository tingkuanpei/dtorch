# DTorch

**DTorch** 是一个基于 **Single-Controller** 与 **Distributed Tensor** 架构的分布式深度学习 API，为 PyTorch 提供高性能分布式计算能力。用户可以在**不修改代码逻辑**的前提下，将单卡 PyTorch 程序扩展到多卡分布式环境。

- **语言**：C++（核心引擎）、Python（框架层与用户接口）
- **计算后端**：LibTorch（PyTorch C++ 算子库）
- **通信**：NCCL（GPU 集合通信）、ZMQ（节点间消息传递）、gRPC（集群 RPC）
- **构建系统**：CMake + scikit-build + nanobind

---

## 三大核心设计理念

### Single-Client Single-Controller Multi-Worker

异步分布式执行模型：Python Client 创建 DTensor 与 Operator 并异步发送，C++ MainNode 统一构建计算图、管理全部计算资源，每个 Worker 绑定一个 CUDA Stream 按序执行 Kernel。

→ [设计理念](developer_guide/design_concept.md) · [Single-Controller 架构](developer_guide/single_controller.md)

### Distributed Tensor (DTensor)

通过 `DeviceMesh` 与 `Placements`（Shard / Replicate / Partial）原生支持 N-D 并行，统一表达 Data Parallel、Tensor Parallel、Context Parallel、Pipeline Parallel、Expert Parallel、ZeRO 等所有并行策略。

→ [Distributed Tensor](user_guide/distributed_tensor.md)

### Eager Graph Architecture

融合 Eager Mode（简洁接口）与 Graph Mode（全局优化）的四层执行引擎：计算图表示 → Kernel 运行时 → Eager Graph 引擎 → 集合通讯组件，三级异步流水线，仅取值时同步。

→ [Eager Graph 架构](developer_guide/eager_graph_architecture/eager_graph_architecture.md)

---

## 快速链接

| 我想… | 去哪里 |
|---|---|
| 构建项目 | [构建指南](get_started/how_to_build.md) |
| 跑测试 | [测试指南](get_started/test.md) |
| 了解 Python API 与 DTensor | [Python API](user_guide/python_api.md) |
| 浏览整体架构 | [总体架构](developer_guide/architecture.md) |
