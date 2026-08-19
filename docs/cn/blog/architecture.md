# DTorch 架构设计：简洁与高效何以兼得

DTorch 是一套基于 Single-Controller 和 Distributed Tensor 架构的分布式深度学习 API，目标是让用户在不修改代码逻辑的前提下，将单卡 PyTorch 程序扩展到多卡分布式环境。本文介绍 DTorch 架构背后的设计动机，以及支撑它的三大核心设计。

前置阅读：

- [Single-Controller 与 Multi-Controller](https://tingkuanpei.github.io/dtorch/cn/developer_guide/single_and_multi_controller/)——两种控制范式的概念介绍
- [Distributed Tensor Overview](https://tingkuanpei.github.io/dtorch/cn/user_guide/distributed_tensor_overview/)（Distributed Tensor 概述）

## 1. 问题：Single-Controller 的调度开销

Single-Controller 通过跨机网络控制分布式 GPU 集群，而 Multi-Controller 中每台机器的 Controller 只需经过 PCIe 总线即可控制本机 GPU。因此，Single-Controller 的调度开销高于 Multi-Controller。但这一开销换来的是整个集群的全局视角和更简洁的编程模型；如何将其摊销，是 DTorch 架构设计需要解决的核心问题。

深度学习程序的特点恰好提供了摊销的空间：程序通常由一系列 Tensor 和 Operator 组成，输入/输出 Tensor 的数据类型、Shape 等元信息在 Operator 执行计算前即可确定，仅在极少数情况下（如 [torch.nonzero](https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html) 和 [torch.Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html)）才会基于输出 Tensor 的值进行分支跳转。因此可以提前构造计算图，让构图的耗时与分布式系统的调度、Kernel 的计算相互重叠（overlap），从而降低系统的调度与运行开销，提升程序性能。

DTorch 正是围绕这一洞察构建了一套简洁且高效的深度学习 API，其架构由三大设计支撑：

1. **Single-Client Single-Controller Multi-Worker** — 异步分布式执行模型
2. **Distributed Tensor** — 原生多维分布式张量抽象
3. **Eager Graph Architecture** — 融合 Eager Mode 与 Graph Mode 优势的执行引擎

下文逐一展开。

## 2. Single-Client Single-Controller Multi-Worker

用户使用 DTorch API 时，只需在单线程 Python 代码中用 DTensor 描述计算（创建 Tensor、调用 Operator），框架便会自动完成分布式集群上的资源管理、任务分发与通讯等操作。DTorch 中有三类角色：Client、Controller 和 Worker，构成 Single-Client Single-Controller Multi-Worker 架构，如下图所示。

<figure markdown>
  ![Single-Client Single-Controller Multi-Worker 架构](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/client_controller_worker.png)
  <figcaption>图 1：Client、Controller 和 Worker 并发执行，并通过 Queue 异步通信</figcaption>
</figure>

**Single-Client** 即 Python Client。用户在 Python Client 中创建 Tensor、调用 Operator、获取 Tensor 的值，这些操作都会被抽象为"计算节点"，序列化成消息（Messages）异步发送给 Controller。Python Client 不会直接创建 CUDA Memory，也不会直接 Launch CUDA Kernel。

Client 侧的 Tensor 只是一个 Symbol，持有 Shape、DType、DeviceMesh 和 Placements 等元信息，不包含实际的数据指针。当对 Tensor 执行计算（如 `tensor_c = tensor_a + tensor_b`）时，Client Thread 会根据输入 Tensor 的 Shape、DeviceMesh 等信息，直接推导出输出 Tensor 的 Shape、DeviceMesh 等信息。

**Client 和 Controller 异步执行：绝大多数情况下，Python Client 无需等待 Controller 即可完成全部计算节点的创建，这极大地降低了分布式系统的调度开销。**只有当后续代码依赖 Tensor 的值（如根据值进行分支跳转、获取并打印 Tensor 的值）时，Client 才需要等待 Controller 完成计算并返回结果。为了减少这类阻塞，DTorch 还提供了 TensorFuture 机制，以异步获取 Tensor 的值。

**Single-Controller** 负责管理所有计算资源：接收 Client 发送的消息，将计算节点组织为计算图；再根据计算图完成 Tensor 的创建与释放、CUDA Kernel 的创建与调度、CUDA Stream 的创建与同步、Communicate Group 的管理等操作。分布式集群中有且仅有一个 Main-Controller，运行在主节点上；每台机器上还有一个 Sub-Controller，负责与 Main-Controller 通讯，并根据其指令直接管理本机资源。Controller 会将计算节点翻译为可执行的 Kernel，发送给 Worker 执行。**Controller 和 Worker 同样异步执行，只有获取 Tensor 的值时才需要同步。**

**Multi-Worker** 负责执行计算任务。每个 Worker 都是一个 C++ Thread，GPU Worker 还会持有一个 CUDA Stream。Worker 依次执行 Controller 分发过来的 Kernel，完成计算。一块 GPU 可以对应多个计算 Worker 和通讯 Worker，实现计算和通讯的重叠。同一个 Tensor 可能同时被同一台机器上的多个 Worker 读写，Controller 会自动插入必要的同步节点（CUDA Event），避免发生 Multi-Thread Race Condition。

## 3. Distributed Tensor

**DTorch 原生支持 DTensor，并基于 DTensor 构建了完善且易用的分布式 API。**PyTorch 中同样支持 DTensor，但目前仍处于 ["alpha state and under development"](https://docs.pytorch.org/docs/stable/distributed.tensor.html)。DTensor 通过 DeviceMesh 和 Placements（Replicate、Shard、Partial）表示 N-D 并行，支持目前所有的并行方式，包括 Data Parallel、Tensor Parallel、Context Parallel、Pipeline Parallel、Expert Parallel 和 ZeRO。

相比 PyTorch SPMD 风格的普通 Tensor，使用 DTensor 无需手动切分或聚合各设备上的 Tensor，代码简洁且符合直觉。例如用 PyTorch SPMD 的 Tensor 实现 Tensor Parallel 时，需要显式切分 Linear 的权重；计算过程中需要聚合结果时，还必须手动调用 all-gather 等集合通讯算子，代码繁琐且极易出错。

### 3.1 算子原生支持 DTensor

DTorch 支持直接创建 DTensor（提供 `randn`、`ones`、`empty` 等 Tensor 创建接口），且所有 Operator 均原生支持 DTensor——和单机 Tensor 一样，直接调用所需的 Operator 即可。每个 Operator 都有一系列 `PlacementSignature` 规则，用于描述输入/输出 Placements 的对应关系。框架会依据 `PlacementSignature` 及其他规则，从输入 Tensor 的 DeviceMesh 和 Placements 自动推导出输出 Tensor 的 DeviceMesh 和 Placements。

当输入 Tensor 的 DeviceMesh 或 Placements 无法满足算子的 `PlacementSignature` 时，需要插入 `tensor.redistribute()` 操作，将其转换成算子要求的分布方式。由于此操作通常很耗时，默认情况下 DTorch 不会自动插入，而是报错提示用户修改代码。为了便于使用，少数算子会自动插入 `redistribute()`（这类算子会在文档中单独说明），例如：Shard Tensor 与 Replicate Tensor 相加、Shard Tensor 调用 `tensor.sum()`、scaled dot-product attention 支持 Ulysses 与 Ring Context Parallel 等。

### 3.2 通讯即算子

DTorch 通过 `tensor.redistribute()` 改变 Tensor 的 DeviceMesh 和 Placements。用户不需要显式创建并管理 [ProcessGroup](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)，也不需要手动调用 [all_reduce](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) 等集合通讯算子——DTorch 中的 DeviceMesh 仅仅表示一组 Device ID，框架在底层创建、管理并调用对应的 ProcessGroup。

在 PyTorch 中，集合通讯必须显式指定 ProcessGroup，并隐式运行在其绑定的 CUDA Stream 上；而在 DTorch 中，`redistribute()` 仅仅是一个普通算子，底层可以根据计算图及网络拓扑选择最高效的实现，并实现计算和通讯的重叠——在确保接口易用的同时，获得更好的性能。DTorch 还支持 DTensor 的不均匀切分：由于 NCCL 的部分集合通讯 primitive 要求所有 rank 上输入 Tensor 的 Shape 完全一致，`redistribute()` 会按需自动补齐或移除 padding。

### 3.3 自由组合的并行方案

这些并行方式的实现方式不尽相同：Data Parallel 和 Pipeline Parallel 只需配置正确的 DeviceMesh 和 Placements；Tensor Parallel 提供了 `ColumnParallelLinear`、`RowParallelLinear` 和 `EmbeddingWithReplicateOutput` 等并行 Module，方便用户调用；Context Parallel 则通过 DeviceMesh 中的 `ulysess_cp` / `ring_cp` 命名维度启用，`scaled_dot_product_attention` 检测到相应维度后，会自动切换到 Ulysses 或 Ring 对应的 CP 实现。各并行方式在 Module 层的具体用法见 [Module 并行](https://tingkuanpei.github.io/dtorch/cn/user_guide/module_parallel/)。

PyTorch 受限于对既有 Module 代码的兼容，只能通过 [parallelize_module](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module) 给 Module 增加 hook 的方式实现并行；DTorch 则直接使用这些并行接口，简单直观，不需要用户理解复杂的 hook 调用逻辑。DTorch 的所有并行方式都可以自由组合，同一份代码既可以单卡运行，也可以组合开启不同的并行方式运行；Llama 模型上 DP + TP + PP + CP 的完整实现见 [Llama 并行示例](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/)。

受益于 Single-Controller 和 DTensor 的架构，开启 Tensor Parallel、Pipeline Parallel 和 Expert Parallel 后，加载模型 state_dict 时无需显式切分并读取对应的模型参数（类似 Megatron-LM 中 TP 和 PP 对 Parameter 的切分），所有切分操作都由框架根据 Tensor 的 DeviceMesh 和 Placements 隐式完成。

## 4. Eager Graph Architecture

为了高效实现 Single-Client Single-Controller Multi-Worker 的范式，DTorch 设计了全新的执行架构：**Eager Graph Architecture**，融合了 Eager Mode 和 Graph Mode 各自的优点。Eager Mode 是 PyTorch 使用的架构，Python 代码直接在 GPU 上创建 Tensor 并 Launch CUDA Kernel，接口简洁易用；Graph Mode 是 TensorFlow v1 使用的架构，先构建计算图再执行计算，可以根据计算图进行全局优化，性能更好，但接口不够直观。

Eager Graph Architecture 是一个基于 Graph 的执行引擎，对外却提供 Eager 接口。用户在 Single-Client 上以 Eager 方式创建计算节点（创建 Tensor、执行 Operator、获取 Tensor 的值等），计算节点被序列化成消息异步发送给 Controller；Controller 根据接收到的消息创建计算子图（并非完整的计算图，而是增量子图），并交给 Graph 引擎执行。由此，DTorch 既保留了 Eager 接口的易用性，又获得了基于子图的全局优化能力。


<figure markdown>
  ![Eager Graph 架构图](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/eager_graph_architecture.png)
  <figcaption>图 2：Eager Graph Architecture 架构图</figcaption>
</figure>

### 4.1 图表示

DTorch 使用 Operand、Operator 和 LogicalGraph 三个抽象表示计算图，其中仅包含计算图的元信息（不包含数据指针、CUDA Kernel 执行代码等）。在 Single-Client 上，Operator 根据输入 Operand 的元信息（DataKind、Device 和 Shape）即可推断出输出 Operand 的元信息，因此计算图可以异步构建。

### 4.2 图执行

参照 CUDA Stream 的范式，DTorch 提出了由 Blob、Kernel 和 Stream 组成的执行引擎：

- **Blob** 表示张量，持有分配好的内存（CPU Memory 或 CUDA Memory）。
- **Kernel** 对输入 Blob 执行计算，并将结果写到输出 Blob 中；其内部是 CPU 代码或 CUDA Kernel，是真正执行计算的单元。
- **Stream** 是计算芯片提供的 CPU 线程、CUDA Stream 等执行资源的统一抽象。Kernel 必须在 Stream 上执行，不同 Stream 之间的计算相互独立、并发执行，Stream 之间支持通过 Event 进行同步。

### 4.3 异步计算与图改写

Eager Graph Architecture 有两个核心优势：**异步计算**和**图改写**。

Single-Client、Single-Controller 和 Multi-Worker 构成异步执行引擎，降低了系统调度的开销。同时，得益于 Single-Client 与 Single-Controller 之间的异步，Controller 可以提前获得计算子图，从而在子图执行前进行图改写，实现计算和通讯重叠、算子融合、冗余节点消除、临时 Tensor 的显存复用等优化；更进一步，还可以接入深度学习编译器进行 JIT 代码生成。

### 4.4 三级并发模型

DTorch 支持三个层次的并行，充分发挥硬件的并发潜能：**Graph 间**、**Operator 间**、**Operator 内**。

**Graph 间并行**：DTorch 中可以创建多个 Graph，不同 Graph 在不同线程上执行。因此，当用户需要并发时，创建多个 Graph 实例即可，不需要手动创建和管理线程。

**Operator 间并行**：在同一个 Graph 内，将不同 Operator 分配到不同的 Stream 上，即可实现 Operator 间的并行。在计算图执行前，DTorch 会根据配置为每个 Operator 分配最优的 Stream。基于此功能，可以实现多线程并发计算、数据传输与计算重叠、多 GPU 并发等能力。

**Operator 内并行**：单个 Operator 内部支持多种并行：

- 将一个 Operator 的计算分发到不同的 Stream 上完成（即 Distributed Tensor 计算）；
- 在 CPU 上使用线程池并行执行 for 循环；
- 利用计算设备提供的 SIMD、SIMT、MIMD 及数据传输与计算重叠的能力。

### 4.5 LibTorch 后端

PyTorch 提供了名为 [LibTorch](https://pytorch.org/cppdocs/) 的 C++ 算子库。DTorch 使用 LibTorch 作为计算后端，大幅降低了算子的开发成本；同时 DTorch 和 PyTorch 调用同一算子库，也天然降低了算子精度对齐的成本。

## 5. 总结

Single-Controller 以跨机网络带来的调度开销，换来了全局视角和更简洁的编程模型；而深度学习程序"元信息可提前确定"的特点，使这笔开销可以通过异步执行来摊销。围绕这一点，DTorch 的架构由三个相辅相成的设计组成：

| 设计 | 解决的问题 | 核心机制 |
|---|---|---|
| Single-Client Single-Controller Multi-Worker | Single-Controller 调度开销高 | Client → Controller → Worker 三级全异步流水线，仅在获取 Tensor 值时同步 |
| Distributed Tensor | 分布式代码繁琐易错 | DeviceMesh + Placements 声明式描述切分，切分、聚合与通讯由框架隐式完成 |
| Eager Graph Architecture | Eager 难以全局优化，Graph 接口不直观 | 对外提供 Eager 接口，内部以增量子图执行，支持异步计算与图改写 |

三者组合的最终效果是：用户以单卡 PyTorch 的写法描述计算，不修改代码逻辑即可扩展到多卡分布式环境。框架在背后完成调度、切分与通讯，并将计算和通讯重叠、算子融合、显存复用等优化建立在提前获得的计算子图之上——简洁与高效由此兼得。

DTorch 的代码和文档均在 [GitHub](https://github.com/tingkuanpei/dtorch) 开源，欢迎关注与参与。

## 延伸阅读

- [关键概念](https://tingkuanpei.github.io/dtorch/cn/developer_guide/key_concept/) — 三大核心设计详解
- [设计决策](https://tingkuanpei.github.io/dtorch/cn/developer_guide/design_decisions/) — 关键设计决策的动机与方案
- [Single-Controller 架构](https://tingkuanpei.github.io/dtorch/cn/developer_guide/single_controller/) / [Distributed Tensor](https://tingkuanpei.github.io/dtorch/cn/developer_guide/distributed_tensor/) / [Eager Graph 架构](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/eager_graph_architecture/)
- [User Guide](https://tingkuanpei.github.io/dtorch/cn/user_guide/user_guide/) — 用 Python API 编写分布式程序
