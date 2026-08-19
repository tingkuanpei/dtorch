# Single-Client Single-Controller Multi-Worker

前置阅读：[Single-Controller 与 Multi-Controller](single_and_multi_controller.md)（两种控制范式的概念介绍）。

DTorch 的分布式执行模型由一个 Python Client、一个中央 Controller 和多个 Worker 组成。Client 负责描述计算节点并异步发送给 Controller，Controller 统一构建计算图并调度资源，Worker 并行执行 Kernel。三者通过异步消息传递流水线式推进，仅在获取 Tensor 值时同步等待，兼顾了编程易用性与分布式执行效率。

## 1. 架构总览

用户使用 DTorch API 时，在单线程的 Python 代码中使用 DTensor 描述计算节点（创建 Tensor、调用
Operator），框架自动完成分布式集群上的资源管理、任务分发和通讯等操作。DTorch 中包含三类角色：**Client**、**Controller** 和 **Worker**，共同构成
Single-Client Single-Controller Multi-Worker 的异步分布式执行模型：

<figure markdown>
  ![Single-Client Single-Controller Multi-Worker 架构](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/client_controller_worker_cn.png)
  <figcaption>图 1：Client、Controller 和 Worker 并发执行，并通过 Queue 异步通信</figcaption>
</figure>

### 1.1 Single-Client

**Single-Client** 即 Python Client 线程。用户在 Python Client 中创建 Tensor、调用 Operator 和获取 Tensor
的值。Python Client 上的所有操作都被抽象为"计算节点"，并序列化为 Messages 异步发送给 Controller。

Python Client **不直接**创建 CUDA Memory 或调用 CUDA Kernel 执行计算。Client 侧的 Tensor 仅仅是一个
**Symbol**，持有 Shape、DType、DeviceMesh 和 Placements 等元信息，但**不持有数据指针**。当执行计算如
`tensor_c = tensor_a + tensor_b` 时，Python Client 根据输入 Tensor 的 Shape、DeviceMesh 等元信息即可直接推断出输出 Tensor 的 Shape、DeviceMesh
等信息，无需等待实际计算完成。

> **Client 与 Controller 异步执行**：绝大多数情况下 Python Client 无需等待 Controller 即可完成所有计算节点的创建，这极大地降低了分布式系统中的通讯开销。仅当 Python Client
> 需要根据 Tensor 的**值**决定后续代码分支时（如 `nonzero`、`.item()` 等），才需等待 Controller 返回计算结果。DTorch 还提供 **Tensor Future**
> 机制，允许异步获取 Tensor 的值，避免主线程阻塞。

### 1.2 Single-Controller

**Single-Controller** 负责管理全部计算资源。在分布式集群中，有且仅有一个 **Main-Controller** 运行在主节点上；每个运行节点上有一个
**Sub-Controller**，负责与 Main-Controller 通讯并根据指令直接管理本节点的资源。由于 Sub-Controller 是 Main-Controller 的附庸，因此将 Main-Controller + Sub-Controller 统称为 Single-Controller。

Controller 接收 Client 发送的计算节点 Messages，将其转换为计算图（LogicalGraph），并基于此完成：

- Tensor 生命期的管理
- CUDA Kernel 创建与调度
- CUDA Stream 创建与同步
- Communicate Group 管理
- 计算图改写优化（算子融合、计算-通信重叠等）

Controller 将计算节点翻译为可执行的 Kernel，分发给 Worker 执行。

> **Controller 与 Worker 同样异步执行**，仅当获取 Tensor 的值时才需同步。

### 1.3 Multi-Worker

**Multi-Worker** 负责执行实际的计算任务。每个 Worker 是一个 C++ Thread，GPU Worker 则额外持有一个 CUDA Stream。Worker
按序执行 Controller 发来的 Kernel，完成计算任务。

关键特性：

- **多 Worker 并发**：一块 GPU 可同时对应多个计算 Worker 与通讯 Worker，实现计算与通讯的**重叠**（overlap）。
- **自动同步**：Tensor 可被同机不同 Worker 并发访问，Controller 会自动插入必要的同步节点（CUDA Event），避免 Multi-Thread Race Condition。

## 2. 三级异步执行流水线

DTorch 的三类角色构成**三级异步执行流水线**，基于[生产者-消费者模式](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem)运行：

```
Client (生产 Operator)  →  Controller (生产 Kernel)  →  Worker (执行 Kernel)
```

每一级都异步于下一级运行，仅当用户显式获取 Tensor 的**值**时才触发同步等待。在目前的深度学习训练和推理应用中，需要基于 Tensor
值进行分支跳转的场景频率较低；即使发生，单次同步的开销也较小，因此整体性能影响可接受。

三级异步流水线带来两个核心收益：

1. **低调度开销**：Client 无需等待 Controller，Controller 无需等待 Worker，各级可以流水线式并行推进。
2. **图改写优化空间**：Controller 可提前获得计算子图，在子图执行前进行优化改写，包括：
    - 计算与通讯重叠
    - 算子融合（Fused Operator）
    - 冗余节点消除
    - 临时 Tensor 的显存复用
    - 接入深度学习编译器进行 JIT 代码生成

## 3. Single-Controller VS Multi-Controller

DTorch 的分布式接口中使用 Single-Controller，而 PyTorch 的分布式接口则是基于 SPMD 的 Multi-Controller。

### 3.1 概念对比

**Single-Controller** 只有一个 Main-Controller 节点，负责管理分布式集群中的**所有** GPU 资源，用户在分布式集群的**全局视角**下进行编程。该模式最早在
TensorFlow v1 中被使用，并在 [Pathway](https://arxiv.org/abs/2203.12533) 中也有论述。

**Multi-Controller** 则创建多个 Controller 进程，每个进程只管理一个 GPU，因此在分布式集群的**局部视角**下编程。所有进程执行同一份代码描述计算流程并直接调度 GPU 资源（SPMD 范式）。

<figure markdown>
  ![Single-Controller vs Multi-Controller](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/single_controller_vs_multi_controller.png)
  <figcaption>图 2：Single-Controller(左图)和 Multi-Controller(右图)的示意图</figcaption>
</figure>

Single-Controller 和 Multi-Controller 是 DTorch 和 PyTorch 最根本的差异。两者对比如下：

| 维度 | Single-Controller | Multi-Controller | 备注 |
|---|---|---|---|
| 编程视角 | 分布式集群的**全局视角** | 分布式集群的**局部视角** | |
| 分布式接口的易用性 | 好 | 差 | |
| 资源管理 | 系统自动管理 | 用户手动管理 | 全局设备管理、设备虚拟化、节点故障自动恢复 |
| 任务切分 | 系统自动切分 | 用户手动切分 | 避免通讯死锁、Auto Parallel |
| 分布式接口的灵活性 | 高 | 极高 | 支持 MPMD 范式 |
| 调度开销 | 低 | 极低 | DTorch 通过异步架构降低了 Single-Controller 的调度开销 |

### 3.2 Single-Controller 的优势

#### 3.2.1 易用性

DTorch 选择 Single-Controller 最大的理由是**易用性**。Single-Controller 可以在分布式集群的全局视角上描述计算内容，这天然地与用户的思维习惯一致。

Single-Controller + DTensor 的方案可以简洁地表示深度学习分布式计算所需的各类并行方案（Data Parallel、Tensor Parallel、Pipeline
Parallel、MoE Parallel、ZeRO，以及调度强化学习训练流程等），极大地降低分布式代码的**开发、修改、维护和调试**成本。

#### 3.2.2 全局优化能力

Single-Controller 提供了计算设备（CPU、GPU）和计算图的**全局视野**，可以实现：

- 全局设备管理与设备虚拟化
- 节点故障自动恢复
- 通讯死锁自动避免
- Auto Parallel（自动并行策略搜索与分配）
- JIT Compilation（`torch.compile` 等）

这些能力在 Multi-Controller 的 SPMD 范式下需要大量手动协调或根本无法实现。

### 3.3 Single-Controller 的劣势与应对

#### 3.3.1 调度开销

Multi-Controller 在每台机器上均有 Controller 节点，其调度 GPU 只需经过 PCIe 总线，延迟极低。Single-Controller
只存在一个 Main-Controller，其调度远端机器的 GPU 需要经过**跨机器网络通讯**，因此调度开销大于 Multi-Controller。

**DTorch 的应对**：采用 Single-Client Single-Controller Multi-Worker 的**异步架构**。Client、Controller 和 Worker
间通过生产者-消费者模式异步执行，当且仅当 Python Client 需要根据 Tensor 的值决定后续代码分支时才会发生阻塞。在深度学习训练和推理的典型场景中，这类同步需求出现频率低、单次开销小，因此整体性能影响可接受。

#### 3.3.2 灵活性

Multi-Controller 中每个进程上执行的代码可以完全不一样，因此支持 **MPMD**（Multi Program Multi Data）范式，灵活性极高。理论上，任何需要并行编程的代码均可使用 MPMD 范式实现。

**实际情况**：目前大模型已收敛至 Transformer 架构，MPMD 范式过高的灵活性并未带来实际收益，反而其**易用性差**的劣势正在凸显。Single-Controller 在保持足够灵活性的同时，提供了显著更好的开发体验。

## 4. 源码实现

DTorch 中使用 [Eager Graph 架构](eager_graph_architecture/eager_graph_architecture.md) 实现了 Single-Client Single-Controller Multi-Worker。但是并没有直接使用 Client、Controller、Worker 作为类名，其对照关系如下：

| 角色 | 代码中的对照 |
|---|---|
| Single-Client | 用户构建计算节点时的 Python 线程。|
| Single-Controller | [class EagerGraphExecutor](https://github.com/tingkuanpei/dtorch/blob/main/dtorch/core/graph/eager_graph_executor.h)  |
| Multi-Worker | [class KernelStream](https://github.com/tingkuanpei/dtorch/blob/main/dtorch/core/kernel_stream/kernel_stream.h) |
