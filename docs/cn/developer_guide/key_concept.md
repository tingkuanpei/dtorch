# 关键概念

DTorch 采用 Single-Controller 架构，通过跨机网络传输控制分布式 GPU 集群。相比之下，Multi-Controller 架构中存在多个 Controller，各自仅需经过 PCIe 总线控制本机 GPU，因此 Single-Controller 的调度开销高于 Multi-Controller，但换来了全局视图和更简洁的编程模型。

> DTorch 采用 Single-Controller 架构的设计动机详见 [设计决策](design_decisions.md)

深度学习程序通常由一系列 Tensor 和 Operator 组成，输入/输出 Tensor 的数据类型、Shape 等信息在 Operator 执行计算前即可确定。仅在极少数情况下（如 [torch.nonzero](https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html) 和 [torch.Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html) 等）才需要基于输出 Tensor 的值进行分支跳转。基于这一特点，DTorch 可以提前构造由 Tensor 和 Operator 组成的计算图，从而降低调度与运行开销，提升程序性能。

DTorch 基于上述特点，围绕三大核心概念构建了一套简洁且高效的深度学习 API：

1. **Single-Client Single-Controller Multi-Worker** — 异步分布式执行模型
2. **Distributed Tensor** — 原生多维分布式张量抽象
3. **Eager Graph Architecture** — 融合 Eager Mode 与 Graph Mode 优势的执行引擎

## Single-Client Single-Controller Multi-Worker

用户在单线程 Python（**Client**）中用 DTensor 描述计算（创建 Tensor、调用 Operator），框架自动完成分布式集群上的资源管理、任务分发与通讯。其间涉及三类角色：

```
┌──────────┐       async messages       ┌────────────┐       kernel queue        ┌────────┐
│ Client   │ ─────────────────────────> │ Controller │ ──────────────────────-─> │ Worker │
│ (Python) │                            │ (C++)      │                           │ (C++)  │
│          │ <── sync when get value ── │            │ <─ sync when get value -─ │        │
└──────────┘                            └────────────┘                           └────────┘
```

- **Client**：用户侧的 Python。所有操作被抽象为"计算节点"并异步发送给 Controller；Client 侧的 Tensor 只是一个持有元信息（Shape、DType、DeviceMesh、Placements）的 Symbol，不持有数据，也不直接 Launch CUDA Kernel。
- **Controller**：唯一的全局管理者，接收计算节点、构建计算图，并完成 Tensor / Kernel / Stream 的创建调度与通讯管理。
- **Worker**：实际执行者（C++ 线程，GPU Worker 额外持有 CUDA Stream），按序执行 Controller 分发的 Kernel。

三级之间**全异步**——Client → Controller → Worker 无需互相等待，仅在需要获取 Tensor 的值时才同步。这正是 DTorch 低通讯开销的来源。

详见 [Single-Controller 文档](single_controller.md)。

## Distributed Tensor

DTorch 原生支持 DTensor，通过 `DeviceMesh` 和 `Placement` 表示 N-D 并行，覆盖 Data Parallel、Tensor Parallel、Context Parallel、Pipeline Parallel、Expert Parallel、ZeRO 等所有主流范式。

相比 PyTorch SPMD 风格，DTensor 无需手动切分或聚合各设备上的 Tensor：算子自动推导输出的分布，加载权重与取值时也自动完成切分和聚合，代码更简洁、更符合直觉。

详见用户指南 [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md) 与 [Distributed Tensor](distributed_tensor.md)。

## Eager Graph Architecture

为高效实现 Single-Client Single-Controller Multi-Worker，DTorch 设计了 **Eager Graph Architecture**，融合两种执行模式各自的优点：

- **Eager Mode**（PyTorch）：Python 直接创建 Tensor 并 Launch Kernel，接口简洁，但缺乏全局优化空间。
- **Graph Mode**（TensorFlow v1）：先构建完整计算图再执行，可全局优化，但接口不够直观。

Eager Graph Architecture 对外暴露 Eager 接口，内部却以 Graph 执行：Client 以 Eager 方式产生的计算节点被异步发给 Controller，Controller 将其构建为**计算子图**（增量子图，而非一次性整图）后由 Graph 引擎执行。由此既保留了 Eager 的易用性，又获得了基于子图的全局优化能力（计算-通讯重叠、算子融合、显存复用等）。

详见 [Eager Graph Architecture 文档](eager_graph_architecture/eager_graph_architecture.md)。
