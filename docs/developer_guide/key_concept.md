# 关键概念

DTorch 采用 Single-Controller 架构，通过跨机网络传输控制分布式 GPU 集群。相比之下，Multi-Controller 架构中存在多个 Controller，各自仅需经过 PCIe 总线控制本机 GPU，因此 Single-Controller 的调度开销高于 Multi-Controller，但换来了全局视图和更简洁的编程模型。

> DTorch 采用 Single-Controller 架构的设计动机详见 [设计决策](design_decisions.md)

深度学习程序通常由一系列 Tensor 和 Operator 组成，输入/输出 Tensor 的数据类型、Shape 等信息在 Operator 执行计算前即可确定。仅在极少数情况下（如 [torch.nonzero](https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html) 和 [torch.Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html) 等）才需要基于输出 Tensor 的值进行分支跳转。基于这一特点，DTorch 可以提前构造由 Tensor 和 Operator 组成的计算图，从而降低调度与运行开销，提升程序性能。

DTorch 基于上述特点，围绕三大核心概念构建了一套简洁且高效的深度学习 API：

1. **Single-Client Single-Controller Multi-Worker** — 异步分布式执行模型
2. **Distributed Tensor** — 原生多维分布式张量抽象
3. **Eager Graph Architecture** — 融合 Eager Mode 与 Graph Mode 优势的执行引擎

## Single-Client Single-Controller Multi-Worker

用户使用 DTorch API 时，在单线程的 Python 代码中使用 DTensor 描述计算节点（创建 Tensor、调用 Operator），框架自动完成分布式集群上的资源管理、任务分发和通讯等操作。DTorch 中包含三类角色：Client、Controller 和 Worker，构成 Single-Client Single-Controller Multi-Worker 的架构：

```
┌──────────┐       async messages       ┌────────────┐       kernel queue        ┌────────┐
│ Client   │ ─────────────────────────> │ Controller │ ──────────────────────-─> │ Worker │
│ (Python) │                            │ (C++)      │                           │ (C++)  │
│          │ <── sync when get value ── │            │ <─ sync when get value -─ │        │
└──────────┘                            └────────────┘                           └────────┘
```

**Single-Client** 即 Python Client。用户在 Python Client 中创建 Tensor、调用 Operator 和获取 Tensor 的值。Python Client 上的所有操作都被抽象为”计算节点”，并序列化为 Messages 异步发送给 Controller。Python Client **不直接**创建 CUDA Memory 或调用 CUDA Kernel 执行计算。Client 侧的 Tensor 仅仅是一个 Symbol，持有 Shape、DType、DeviceMesh 和 Placements 等元信息，但不持有数据指针。当执行计算如 `tensor_c = tensor_a + tensor_b` 时，Python Client 根据输入 Tensor 的 Shape、DeviceMesh 等元信息即可直接推断出输出 Tensor 的 Shape、DeviceMesh 等信息。Client 侧的 Operator 同样仅持有参数，不持有 CPU/CUDA Kernel，不负责实际计算。

> **Client 与 Controller 异步执行**，绝大多数情况下 Python Client 无需等待 Controller 即可完成所有计算节点的创建，这极大地降低了分布式系统中的通讯开销。仅当 Python Client 需要根据 Tensor 的值决定后续代码分支时，才需等待 Controller 返回计算结果。DTorch 还提供 Tensor Future 机制，允许异步获取 Tensor 的值，避免主线程阻塞。

**Single-Controller** 负责管理全部计算资源，核心实现位于 `dtorch/core/distributed/main_node.h` 和 `dtorch/core/graph/eager_graph_executor.h`。它接收 Client 发送的计算节点 Messages，将其转换为计算图（LogicalGraph），并基于此完成：Tensor 创建与释放、CUDA Kernel 创建与调度、CUDA Stream 创建与同步、Communicate Group 管理等。在分布式集群中，有且仅有一个 Main-Controller 运行在主节点上；每台机器上有一个 Sub-Controller（`dtorch/core/distributed/worker_node.h`），负责与 Main-Controller 通讯并根据指令直接管理本机资源。Controller 将计算节点翻译为可执行的 Kernel，分发给 Worker 执行。

> **Controller 与 Worker 同样异步执行**，仅当获取 Tensor 的值时才需同步。

**Multi-Worker** 负责执行计算任务。每个 Worker 是一个 C++ Thread，GPU Worker 则额外持有一个 CUDA Stream。Worker 按序执行 Controller 发来的 Kernel，完成计算任务。一块 GPU 可同时对应多个计算 Worker 与通讯 Worker，实现计算与通讯的重叠。Tensor 可被同机不同 Worker 并发访问，Controller 会自动插入必要的同步节点（CUDA Event），避免 Multi-Thread Race Condition。

详见 [Single-Controller 文档](single_controller.md)。

## Distributed Tensor

**DTorch 原生支持 DTensor，并基于 DTensor 构建了完善且易用的分布式 API。**PyTorch 中虽也支持 DTensor，但至今仍处于 [alpha 阶段](https://docs.pytorch.org/docs/stable/distributed.tensor.html)。DTorch 中的 DTensor 通过 `DeviceMesh`（`dtorch/api/cpp/distributed_spec.h`）和 `Placement`（Replicate、Shard、Partial）表示 N-D 并行，支持目前所有主流并行范式：Data Parallel、Tensor Parallel、Context Parallel、Pipeline Parallel、Expert Parallel 和 ZeRO。

相比 PyTorch SPMD 风格的 Tensor，使用 DTensor 无需手动切分或聚合各设备上的 Tensor，代码更简洁且符合直觉。例如使用 PyTorch SPMD 实现 Tensor Parallel 时，需显式切分 Linear 的权重；在计算过程中需要获取 Tensor 值时，还必须手动调用 `all-gather` 等集合通讯算子，代码极易出错。

### 计算

DTorch 支持直接创建 DTensor（提供 `randn`、`ones`、`empty` 等创建接口），且**所有 Operator 均原生支持 DTensor**，与单机 Tensor 一样直接调用即可。每个 Operator 定义一组 `PlacementSignature`（`dtorch/core/operators/placement_signature.h`），描述输入输出 Placement 的对应关系。框架根据 PlacementSignature 及其他规则，基于输入 Tensor 的 DeviceMesh 和 Placements 自动推导输出 Tensor 的 DeviceMesh 和 Placements。

当 DeviceMesh 或 Placements 无法匹配时，需插入 `tensor.redistribute()` 操作。由于此操作通常耗时较大，DTorch 默认不自动执行，而是报错提示用户修改代码。为便于使用，某些常用算子会自动插入 `tensor.redistribute()`，这些算子会在文档中单独说明，例如：
- Shard Tensor 与 Replicate Tensor 相加
- Shard Tensor 调用 `tensor.sum()`
- Scale Dot Product Attention 支持 Ulysses 和 Ring Context Parallel

### 通讯

DTorch 通过 `tensor.redistribute()` 函数改变 Tensor 的 DeviceMesh 和 Placements。用户无需显式创建并管理 [ProcessGroup](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)，也无需调用 [all_reduce](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) 等集合通讯算子。DTorch 中的 DeviceMesh 仅表示一组 Device ID，框架在底层（`dtorch/core/communication/`）自动创建和管理对应的 ProcessGroup。

不同于 PyTorch 中通讯需通过 ProcessGroup 隐式绑定某个 CUDA Stream，`tensor.redistribute()` 本身是一个普通算子，因此 DTorch 底层可根据计算图和网络拓扑选择最高效的实现，并自动实现计算与通讯的重叠 — 在保证接口易用的同时获得更好性能。

DTorch 还支持 DTensor 的**不均匀切分**，在 `tensor.redistribute()` 时自动补 padding 和移除 padding（NCCL 中部分集合通讯 primitive 要求所有 rank 上输入 Tensor 的 shape 完全一致）。

详见 [Distributed Tensor 文档](distributed_tensor.md)。

## Eager Graph Architecture

为高效实现 Single-Client Single-Controller Multi-Worker 的范式，DTorch 设计了全新的执行架构：**Eager Graph Architecture**，融合了 Eager Mode 和 Graph Mode 各自的优点：

- **Eager Mode**（PyTorch 架构）：Python 代码直接在 GPU 上创建 Tensor 并 Launch CUDA Kernel，接口简洁易用，但缺乏全局优化空间。
- **Graph Mode**（TensorFlow v1 架构）：先构建完整计算图再执行，可全局优化，但接口不够直观。

Eager Graph Architecture 基于 Graph 实现的执行引擎，对外暴露 Eager 接口。用户在 Single-Client 以 Eager 方式创建计算节点（创建 Tensor、执行 Operator、获取 Tensor 值等），这些节点被序列化为 Messages 异步发给 Controller。Controller 将接收到的 Messages 构建为**计算子图**（每次构图并非完整计算图，而是增量子图），并通过 Graph 引擎执行。Eager Graph Architecture 既保留了 Eager 接口的易用性，又获得了基于子图的全局优化能力，兼备两种模式的优点。

Eager Graph Architecture 采用四层分层设计：

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

- **Layer 1 — 计算图表示**: Operand（张量元信息节点）、Operator（计算节点）、LogicalGraph（DAG 容器）。仅含元信息，不含数据指针或 CUDA Kernel。
- **Layer 2 — Kernel 运行时**: Blob（torch::Tensor 物理容器）、Kernel（最小执行单元）、KernelStream（CPU 线程/CUDA Stream 封装）。Operator → Kernel 映射，异步执行。
- **Layer 3 — GraphExecutor**: GraphConstructor（Python API 桥梁）→ EagerGraphExecutor（Controller 实现，AsyncMain 消息循环）→ NodeRunnerBase 派生类（NaiveRunner 执行引擎）。支持单机多线程和单机多进程两种模式。
- **Layer 4 — 集合通讯组件**: ThreadGroup（AllReduce/AllGather 等集合通信原语）+ TensorStore（生产者-消费者跨线程张量交换）。支撑 DTensor 的 DeviceMesh 变换和 Placement 重分布。
- **三级异步流水线**: Client → Controller → Worker，仅取值时同步
- **图改写优化**: 计算-通信重叠、算子融合、冗余消除、显存复用

详见 [Eager Graph Architecture 文档](eager_graph_architecture/eager_graph_architecture.md)。
