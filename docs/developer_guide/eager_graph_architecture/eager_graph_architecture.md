# Eager Graph 架构

DTorch 的 **Eager Graph Architecture** 是用于实现 [Single-Client Single-Controller Multi-Worker](../single_controller.md) 异步分布式执行模型的核心引擎，同时也是一个完整的深度学习计算框架运行时。它融合了 **Eager Mode**（PyTorch 风格，接口简洁）和 **Graph Mode**（TensorFlow v1 风格，可全局优化）各自的优势——对外暴露 Eager 接口，内部基于计算子图进行优化和执行。

## 设计动机

深度学习程序通常由一系列 Tensor 和 Operator 组成。输入/输出 Tensor 的数据类型、Shape 等信息在 Operator 执行计算前即可确定，仅在极少数情况下（如 `nonzero`、`.item()`）才需要基于输出 Tensor 的**值**进行分支跳转。基于这一特点，DTorch 可以在 Client 侧仅凭元信息异步构建计算图，无需等待实际计算完成。

Eager Graph Architecture 的设计目标：

- **Eager 接口**：用户在 Python 中以 imperative 风格逐行编写代码（创建 Tensor、调用 Operator），与 PyTorch 体验一致
- **Graph 执行**：Controller 将 Client 发送的算子序列构建为增量子图，在子图执行前进行优化改写
- **异步流水线**：Client → Controller → Worker 三级异步执行，仅获取 Tensor 值时同步等待
- **原生分布式**：通过 DTensor 的 DeviceMesh 和 Placements，统一表达 Data Parallel、Tensor Parallel、Pipeline Parallel 等所有并行策略

---

## 四层架构

Eager Graph Architecture 采用分层设计，从逻辑表示到底层通信形成完整技术栈：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           GraphExecutor (Layer 3)                            │
│  GraphConstructor → EagerGraphExecutor → NodeRunnerBase (具体子类)           │
│  消息队列驱动，异步消费 Operator，管理计算图生命周期                         │
│                                                                              │
│  单机多线程：PerDeviceThreadNodeRunner → NaiveRunner (kMemory store)         │
│  单机多进程：PerDeviceProcessNodeRunner → RemoteRunnerInProcess → GPU 子进程 │
├──────────────────────────────────────────────────────────────────────────────┤
│              计算图表示 (Layer 1)              Kernel 运行时 (Layer 2)       │
│  Operand / Operator / LogicalGraph            Blob / Kernel / KernelStream   │
│  DAG 拓扑结构，元信息推导                      物理内存容器，异步执行流      │
├──────────────────────────────────────────────────────────────────────────────┤
│                            集合通讯组件 (Layer 4)                            │
│  ThreadGroup (集合通信原语)  /  TensorStore (跨线程张量交换)                 │
│  AllReduce, AllGather, ReduceScatter ...       Memory / File / Network       │
└──────────────────────────────────────────────────────────────────────────────┘
```

| 层 | 文档 | 职责 |
|---|---|---|
| **Layer 1: 计算图表示** | [logical_graph_representation.md](logical_graph_representation.md) | 用 Operand（数据节点）和 Operator（计算节点）构建 DAG，表达计算逻辑的元信息 |
| **Layer 2: Kernel 运行时** | [kernel_runtime.md](kernel_runtime.md) | 将 Operator 转化为可执行的 Kernel，通过 Blob 管理物理内存，在 KernelStream 上异步执行 |
| **Layer 3: GraphExecutor** | [graph_executor.md](graph_executor.md) + [graph_executor_in_cluster.md](graph_executor_in_cluster.md) | 消息驱动的执行管线：GraphConstructor → EagerGraphExecutor → NodeRunnerBase（具体子类），支持单机多线程和单机多进程两种模式 |
| **Layer 4: 集合通讯组件** | [tensor_communicate.md](../communicate/tensor_communicate.md) | 提供 ThreadGroup（集合通信）和 TensorStore（跨线程张量交换），支撑分布式 DTensor 的 Placement 变换 |

---

## Layer 1: 计算图表示

**文档**: [logical_graph_representation.md](logical_graph_representation.md)

计算图表示层定义了三个核心抽象，构成 DAG（有向无环图）以拓扑结构表达计算逻辑：

```
Operand (数据节点)                Operator (计算节点)              LogicalGraph (DAG 容器)
┌─────────────────┐              ┌─────────────────┐             ┌──────────────────────┐
│ mShape          │              │ mInputOperands  │             │ mOperatorMap          │
│ mStride         │              │ mOutputOperands │             │ mOperandMap           │
│ mDataKind       │    输入      │ OpParam         │    管理     │                       │
│ mDeviceMesh     │ ──────────►  │ Infer()         │ ──────────► │ AddOperator()         │
│ mPlacementSeq   │              │ Compute()       │             │ DeleteOperator()      │
│                 │ ◄────────── │                 │             │                       │
│ mProducerOp     │    输出      │                 │             │                       │
│ mConsumerOps    │              └─────────────────┘             └──────────────────────┘
└─────────────────┘
```

- **Operand** — 数据节点，持有张量的元信息（Shape、DType、DeviceMesh、Placements），**不持有实际数据**。通过 `mProducerOp` 和 `mConsumerOps` 与 Operator 建立双向拓扑连接。
- **Operator** — 计算节点，封装单个算子（如 `relu`、`add`、`matmul`）的完整生命周期。通过 `Infer()` 根据输入 Operand 的元信息推导输出 Operand 的元信息，使 Client 侧无需等待实际计算即可异步构图。
- **LogicalGraph** — DAG 拓扑容器，管理所有 Operand 和 Operator 的映射关系。

`GraphConstructor`（`dtorch/core/graph/graph_constructor.h`）是 Python API 与核心引擎之间的**桥梁**——接收用户 API 调用，通过 `OperatorFactory` 创建 Operator 并注入执行引擎的消息队列。

---

## Layer 2: Kernel 运行时

**文档**: [kernel_runtime.md](kernel_runtime.md)

Kernel 运行时层负责将 Layer 1 中构建的 Operator 转化为可在具体设备上执行的 Kernel，并分配到对应的 KernelStream 上异步执行：

```
Operator::Infer()
  │
  └─→ InferOperatorAssignInfo()   ──→  OperatorAssignInfo (StreamKeySet)
                                          │
                                          ▼
Runner::CreateKernelForOperator()         ──→  为每个 KernelStreamKey 创建一个 Kernel
  │                                           │
  ├─ 创建/查找 Blob（输入/输出张量容器）        │
  ├─ KernelStreamManager::GetStream()          │
  └─ Kernel::CreateKernel(ctx)                 │
                                          │
                                          ▼
Runner::Execute()
  │
  └─→ kernel->GetStream().LaunchKernel(kernel)  ──→  KernelStream 异步队列
                                                       │
                                                       ▼
                                                KernelStream::AsyncMain()
                                                       │
                                                       └─→ Kernel::Run()
                                                             ├─ 从 Blob 读取输入 torch::Tensor
                                                             ├─ Compute() → Operator::Compute()
                                                             └─ 将输出 torch::Tensor 写回 Blob
```

三个核心组件：

- **Blob** — `torch::Tensor` 的引用计数包装器，是 Kernel 读写张量数据的**物理容器**。分布式场景下，一个 Operand 对应 `AllDeviceBlobs`（按全局设备 ID 索引的 Blob 映射）。
- **Kernel** — 最小执行单元。一个 Operator 在 N 个设备上执行，就对应创建 N 个 Kernel。每个 Kernel 绑定一个 `(Device, StreamType)`，从输入 Blob 读取数据、执行 `Compute()`、将结果写回输出 Blob。
- **KernelStream** — CPU 线程和 CUDA Stream 的封装。每个流绑定一个特定设备的专用线程，通过无锁队列按序执行 Kernel。`KernelStreamType::kCompute` 和 `kCommunicate` 的分离支持计算与通信的重叠。

---

## Layer 3: GraphExecutor

**文档**: [graph_executor.md](graph_executor.md)（单机多线程） + [graph_executor_in_cluster.md](graph_executor_in_cluster.md)（单机多进程）

GraphExecutor 是 Single-Controller 架构中 **Controller** 的实现，负责串联 Layer 1 的图表示和 Layer 2 的 Kernel 运行时。它通过消息队列驱动 `AsyncMain` 线程消费 Operator、构建 LogicalGraph、分发给 Runner 执行。

GraphExecutor 支持两种部署模式，由 `GraphOption::perDevicePerProcess` 控制：

```
单机多线程 (perDevicePerProcess = false):             单机多进程 (perDevicePerProcess = true):
                                                                 Parent Process
GraphConstructor                                        GraphConstructor
      │                                                       │
      ▼                                                       ▼
EGEMessageQueue (同进程)                              EagerGraphExecutor
      │                                                       │
      ▼                                                       ├── RemoteRunnerPublisher (PUB)
EagerGraphExecutor                                           │   广播 Execute / Destroy
      │                                                       │
      ▼                                                       └── PerDeviceProcessNodeRunner
PerDeviceThreadNodeRunner                                         ├── NaiveRunner (CPU)
      │                                                           │
      ▼                                                           └── RemoteRunnerInProcess (per GPU)
NaiveRunner                                                              fork/exec 拉起子进程
      │                                                                    │
      ▼                                       ════════════════════════════╪═══════════════
KernelStream::LaunchKernel()                  Child Process (per GPU)    ▼
      │                                       RemoteRunner
      ▼                                         ├── RemoteRunnerSubscriber (SUB)  ← Execute 广播
CUDA Stream / CPU Thread 执行 Kernel            ├── RemoteRunnerPusher (PUSH: devicesReady)
                                                └── NaiveRunner (GPU)
                                                       │
                                                       ▼
                                                  KernelStream::LaunchKernel()
```

**NodeRunnerBase 统一接口**：`NodeRunnerBase`（`dtorch/core/runner/node_runner_base.h`）是所有 NodeRunner 的抽象基类，仅声明一个纯虚函数 `Execute(ops, noHoldOperands)`。其下各类分工协作覆盖全部场景：

| 类 | 角色 | 实现方式 |
|---|---|---|
| `PerDeviceThreadNodeRunner` | NodeRunnerBase 子类（单机多线程） | 持有 `NaiveRunner`（`kMemory` store），委托执行 |
| `PerDeviceProcessNodeRunner` | NodeRunnerBase 子类（单机多进程） | CPU 持有 `NaiveRunner`；每张 GPU 由 `RemoteRunnerInProcess` 拉起子进程 `RemoteRunner` |
| `NaiveRunner` | 核心执行引擎 | Operator → Kernel 转换、Blob 管理、KernelStream 调度 |
| `RemoteRunner` | 子进程执行引擎 | 子进程内 `NaiveRunner` + `RemoteRunnerSubscriber`(SUB) + `RemoteRunnerPusher`(PUSH) |
| `WorkerNodeMultiGraphNodeRunner` | 多机集群（预留） | 每个 WorkerNode 持有一个 `PerDeviceProcessNodeRunner` |

**单机多线程模式**：[graph_executor.md](graph_executor.md)。`GraphConstructor` 将 Operator 封装为异步 `EGEMessage`，`EagerGraphExecutor::AsyncMain` 消费消息 → 构建 `LogicalGraph` → `PerDeviceThreadNodeRunner` 委托 `NaiveRunner` 将 Operator 转为 Kernel 发射执行。消息队列分两种：异步消息（不阻塞 Producer）和同步消息（仅取 Tensor 值时阻塞）。

**单机多进程模式**：[graph_executor_in_cluster.md](graph_executor_in_cluster.md)。每张 GPU 运行在独立子进程中（`fork`/`exec`），由 `RemoteRunnerInProcess` 拉起并管理生命周期。`EagerGraphExecutor` 持有 `RemoteRunnerPublisher`（PUB）广播 `Execute`/`Destroy`；子进程的 `RemoteRunner` 经 `RemoteRunnerSubscriber`（SUB）接收算子并交由内部 `NaiveRunner` 执行，启动时经 `RemoteRunnerPusher`（PUSH）回报 `devicesReady`。

两种模式对上层 Python API 完全透明。

---

## Layer 4: 集合通讯组件

**文档**: [tensor_communicate.md](../communicate/tensor_communicate.md)

集合通讯组件提供跨设备、跨线程的张量数据传输能力，是 DTensor 实现 DeviceMesh 变换（Scatter、Gather、Redistribute）和 Placement 重分布的基础：

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            Kernel (ConvertKernel / CopyKernel / …)             │
│                                                                               │
│   ┌──────────────────────────┐          ┌─────────────────────────────────┐   │
│   │  ThreadGroup              │          │  TensorStore                     │   │
│   │  (集合通信原语)            │          │  (进程间张量存储)                 │   │
│   │                          │          │                                  │   │
│   │  AllReduce / AllGather   │          │  SrcSet  / DestGet              │   │
│   │  ReduceScatter / AllToAll│          │  Barrier / Reset                │   │
│   │  ReplicateToShard / …   │           │                                  │   │
│   │                          │          │  ┌─ MemoryTensorStore (同进程)   │   │
│   │  ┌────────────────────┐  │          │  ├─ FileTensorStore   (跨进程)   │   │
│   │  │ Backend            │  │          │  └─ NetworkTensorStore(跨机器)   │   │
│   │  ├─ ProcessGroupNCCL  │  │          │                                  │   │
│   │  └─ SameDeviceBackend │  │          │                                  │   │
│   │  └────────────────────┘  │          └─────────────────────────────────┘   │
│   └──────────────────────────┘                                                │
│                                                                               │
│   ThreadGroupManager ──► 查找/创建 ThreadGroup ──► 按 ThreadGroupKey 缓存       │
└───────────────────────────────────────────────────────────────────────────────┘
```

两层通信抽象：

- **TensorStore**（底层）— 生产者-消费者风格的跨线程张量交换。提供 `SrcSet`（写入）/ `DestGet`（读取）API，通过 CUDA Event 而非全局锁实现异步同步。三种后端：
  - `MemoryTensorStore`：同一进程内多线程（基于条件变量 + CUDA Event）
  - `FileTensorStore`：同一机器多进程（基于文件系统 + Boost IPC）
  - `NetworkTensorStore`：跨机器（预留）

- **ThreadGroup**（上层）— 集合通信原语（AllReduce、AllGather、ReduceScatter、AllToAll 等），将一组设备上的线程组成通信组。两种后端自动选择：
  - `ProcessGroupNCCL`：不同 GPU 间的高带宽通信（NVLink/PCIe）
  - `ThreadGroupSameDeviceBackend`：同一 GPU 上的多线程通信（将集合通信翻译为 TensorStore 操作序列，无需 NCCL）

`ConvertKernel` 是使用通信设施最多的 Kernel，负责 DTensor 的 DeviceMesh 迁移和 Placement 重分布（S→R, R→S, S→S, P→R, P→S 等）。

---

## 端到端执行流程

以下以 `c = a + b` 为例，展示一次加法运算经过四层组件的完整路径：

```
Python: c = a + b
    │
    ▼
Layer 3: GraphConstructor::AddOperator()
    ├── Layer 1: OperatorFactory::NewOperatorOrThrow("add", {a.operand, b.operand})
    │   └── Operator::Infer() → 推导输出 Operand 的 Shape/DType/Placements
    ├── SendOperatorToExecutor()
    │   └── EGEMessageQueue::PushMessage(AddOperatorEEMsg{op})  ← 异步，立即返回
    └── 返回 api::cpp::Tensor(c_operand)  ← 仅持有元信息，无数据
    │
    ▼  (异步 — Producer 线程继续，不等待)
    │
Layer 3: EagerGraphExecutor::AsyncMain()  // 消费者线程
    ├── 消费消息 → LogicalGraph::AddOperator(op)
    ├── GraphTraversalSequence 拓扑排序
    └── Runner::Execute(ops)
        │
        ▼
Layer 3 → Layer 2: NaiveRunner::CreateKernelForOperator(add_op)
    ├── Layer 1: 读取 OperatorAssignInfo → StreamKeySet (e.g. {GPU:0, GPU:1})
    ├── Layer 2: 查找输入 Blob (a.blob, b.blob)
    ├── Layer 2: 分配输出 Blob (c.blob)
    ├── Layer 2: mStreamManager.GetStream(device, kCompute)
    └── Layer 2: Kernel::CreateKernel(ctx) → CudaAddKernel
    │
    ▼
Layer 2: KernelStream::LaunchKernel(kernel)  ← 异步发射
    │
    ▼  (KernelStream::AsyncMain 线程)
Layer 2: Kernel::Run()
    ├── 从 Blob 读取 torch::Tensor 输入
    ├── Compute() → Operator::Compute() → LibTorch 算子
    └── 将输出 torch::Tensor 写回 Blob
    │
    ▼  (仅获取 Tensor 值时触发：插入 GetTensorOp)
Layer 3: EagerGraphExecutor → Runner::Execute([get_tensor_op])
    └── GetTensorOp::Compute()
        ├── 从 Blob 读取 c 的 torch::Tensor
        ├── (CUDA) 记录 Event → 后台线程等待 Event 就绪
        └── promise->SetValue(tensor) → 唤醒阻塞在 future.get() 的 Producer 线程
```

---

## 与 Single-Controller 架构的关系

Eager Graph Architecture 是 [Single-Client Single-Controller Multi-Worker](../single_controller.md) 的具体实现：

| 角色 | Eager Graph 中的对应组件 |
|---|---|
| **Single-Client** | 用户构建计算节点时的 Python 线程。通过 `GraphConstructor` 将 Operator 序列化为消息发送。Client 侧 Tensor 仅持有元信息，不持有数据。 |
| **Single-Controller** | `EagerGraphExecutor`（`dtorch/core/graph/eager_graph_executor.h`）。在独立线程中运行 `AsyncMain` 循环，管理 LogicalGraph 生命周期，调度 Runner 执行。 |
| **Multi-Worker** | `KernelStream`（`dtorch/core/kernel_stream/kernel_stream.h`）。每个 Worker 绑定一个 CUDA Stream 或 CPU 线程，按序执行 Kernel 队列。 |

三级异步流水线：

```
Client (生产 Operator)  →  Controller (生产 Kernel)  →  Worker (执行 Kernel)
      异步                       异步
```

仅当用户显式获取 Tensor 的**值**时才触发同步等待，其余阶段全部异步执行，兼顾了编程易用性与分布式执行效率。

DTorch 提供了异步获取 Tensor 值的机制，进一步降低了系统的同步等待开销。详见 [异步获取 Tensor 值](async_get_tensor.md)。

---

## 相关文档

| 文档 | 内容 |
|---|---|
| [design_concept.md](../design_concept.md) | DTorch 三大核心设计理念 |
| [single_controller.md](../single_controller.md) | Single-Client Single-Controller Multi-Worker 架构 |
| [distributed_tensor.md](../distributed_tensor.md) | DTensor 的 DeviceMesh 与 Placements 机制 |
| [logical_graph_representation.md](logical_graph_representation.md) | Layer 1: 计算图表示 — Operand / Operator / LogicalGraph |
| [kernel_runtime.md](kernel_runtime.md) | Layer 2: Kernel 运行时 — Blob / Kernel / KernelStream |
| [graph_executor.md](graph_executor.md) | Layer 3: GraphExecutor — GraphConstructor → EagerGraphExecutor → Runner |
| [tensor_communicate.md](../communicate/tensor_communicate.md) | Layer 4: 集合通讯组件 — ThreadGroup / TensorStore |
| [graph_executor_in_cluster.md](graph_executor_in_cluster.md) | 集群场景：多机多进程执行管线 |
| [zmq.md](../communicate/zmq.md) | ZMQ 通信机制：PUB-SUB + PUSH-PULL 双通道 |
| [python_kernel.md](python_kernel.md) | 在 C++ Kernel 中调用 Python 代码 |
| [async_get_tensor.md](async_get_tensor.md) | 异步获取 Tensor 值：Promise/Future 机制、GetTensorOp、File/Memory 实现 |
