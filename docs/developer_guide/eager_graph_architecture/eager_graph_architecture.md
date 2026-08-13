# Eager Graph 架构

DTorch 的 **Eager Graph Architecture** 是用于实现 [Single-Client Single-Controller Multi-Worker](../single_controller.md) 异步分布式执行模型的核心引擎，同时也是一个完整的深度学习计算框架运行时。它融合了 **Eager Mode**（PyTorch 风格，接口简洁）和 **Graph Mode**（TensorFlow v1 风格，可全局优化）各自的优势——对外暴露 Eager 接口，内部基于计算子图进行优化和执行。

## 设计动机

深度学习程序通常由一系列 Tensor 和 Operator 组成。输入/输出 Tensor 的数据类型、Shape 等信息在 Operator 执行计算前即可确定，仅在极少数情况下（如 `nonzero`、`.item()`）才需要基于输出 Tensor 的**值**进行分支跳转。基于这一特点，DTorch 可以在 Client 侧仅凭元信息异步构建计算图，无需等待实际计算完成。

Eager Graph Architecture 的设计目标：

- **Eager 接口**：用户在 Python 中以 imperative 风格逐行编写代码（创建 Tensor、调用 Operator），与 PyTorch 体验一致
- **Graph 执行**：Controller 将 Client 发送的算子序列构建为增量子图，在子图执行前进行优化改写
- **异步流水线**：Client → Controller → Worker 三级异步执行，仅获取 Tensor 值时同步等待

---

## 四层架构

Eager Graph Architecture 采用分层设计，从逻辑表示到底层通信形成完整技术栈：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           GraphExecutor (Layer 3)                            │
│ GraphConstructor → EagerGraphExecutor → NodeRunnerBase (concrete subclasses) │
│ Message-queue driven; consumes Operators async; manages graph create/destroy │
│                                                                              │
│    Multi-thread: PerDeviceThreadNodeRunner → NaiveRunner (kMemory store)     │
│      Multi-process: EagerGraphExecutor broadcasts via Publisher (PUB);       │
│   PerDeviceProcessNodeRunner → RemoteRunnerInProcess → child RemoteRunner    │
├──────────────────────────────────────────────────────────────────────────────┤
│    Graph Representation (Layer 1)    │       Kernel Runtime (Layer 2)        │
│  Operand / Operator / LogicalGraph   │     Blob / Kernel / KernelStream      │
│  DAG topology, meta-info deduction   │    Memory container, async stream     │
├──────────────────────────────────────────────────────────────────────────────┤
│                      Collective Communication (Layer 4)                      │
│  ThreadGroup (collective primitives) / TensorStore (cross-thread exchange)   │
│       AllReduce, AllGather, ReduceScatter ... Memory / File / Network        │
└──────────────────────────────────────────────────────────────────────────────┘
```

| 层 | 文档 | 职责 |
|---|---|---|
| **Layer 1: 计算图表示** | [logical_graph_representation.md](logical_graph_representation.md) | 用 Operand（数据节点）和 Operator（计算节点）构建 DAG，表达计算逻辑的元信息 |
| **Layer 2: Kernel 运行时** | [kernel_runtime.md](kernel_runtime.md) | 将 Operator 转化为可执行的 Kernel，通过 Blob 管理物理内存，在 KernelStream 上异步执行 |
| **Layer 3: GraphExecutor** | [graph_executor.md](graph_executor.md) + [graph_executor_in_cluster.md](graph_executor_in_cluster.md) | 消息驱动的执行管线：GraphConstructor → EagerGraphExecutor → NodeRunnerBase（具体子类），支持单机多线程和单机多进程两种模式 |
| **Layer 4: 集合通讯组件** | [tensor_communicate.md](../communicate/tensor_communicate.md) | 提供 ThreadGroup（集合通信）和 TensorStore（跨线程张量交换），支撑分布式 DTensor 的 Placement 变换 |
