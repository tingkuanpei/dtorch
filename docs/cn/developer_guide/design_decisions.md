# 设计决策

此文档将描述 DTorch 一系列的关键设计决策。包括为什么选择 DTensor + Single-Controller 的方案，如何解决分布式系统的调度开销，选择 LibTorch 作为后端等等。通过此文档，可以了解 DTorch 架构的设计的意图及方案。

## DTensor

分布式并行要求同一个 Tensor 被切分到多台设备上，各设备之间还需协调完成计算与通讯。若由用户手动管理这一过程，需显式切分参数、手动调用集合通讯算子，代码繁琐且极易出错。为此，DTorch 引入 DTensor，用户通过 `DeviceMesh` 和 `Placements` 声明式地描述 Tensor 的切分方式，由框架自动完成切分、聚合与通讯，同一份代码无需修改即可在单卡或多卡上运行，从而解决分布式并行下代码繁琐易错的问题。

## Single-Controller

PyTorch 的分布式接口基于 SPMD 的 Multi-Controller 方案，每个进程仅管理一个 GPU，用户只能在局部视角下编程，ProcessGroup 的创建、参数的切分、集合通讯的调用均需手动完成；节点故障恢复、通讯死锁避免、Auto Parallel 等全局优化能力在 SPMD 范式下需要大量手动协调或根本无法实现，分布式接口的易用性较差。为此，DTorch 选择 Single-Controller，由唯一的 Controller 在全局视角下自动管理集群中的所有 GPU 资源，从而获得更简洁的编程模型与更强的全局优化能力。

> PyTorch 基于 SPMD 的 Multi-Controller 方案以普通 Tensor 为设计前提；当切换到 DTensor 后，Single-Controller 的方案易用性更好。

## Single-Client Single-Controller Multi-Worker
Single-Controller 通过跨机网络统一调度所有机器，相比每台机器各自通过 PCIe 总线管理本机 GPU 的 Multi-Controller 方案，跨机调度带来了更高的调度开销。针对这一开销，DTorch 采用异步执行机制。深度学习的程序通常由一系列 Tensor 和 Operator 组成，输入/输出 Tensor 的数据类型、Shape 等元信息在 Operator 执行计算前即可确定。基于该特点，DTorch 将计算节点的构造、分布式系统的调度与 Kernel 的计算相互重叠（overlap），形成了**Single-Client Single-Controller Multi-Worker** — 异步分布式执行模型，从而解决 Single-Controller 调度开销大的问题。

## 静态图和动态图统一的计算引擎

深度学习框架的执行模式可分为动态图与静态图两类：动态图（Eager Mode，PyTorch 架构）中 Python 代码直接创建 Tensor 并 Launch CUDA Kernel，接口简洁易用，但缺乏全局优化空间；静态图（Graph Mode，TensorFlow v1 架构）先构建完整计算图再执行，可进行全局优化，但接口不够直观。DTorch 选择将两者统一：执行引擎基于计算图实现，对外暴露 Eager 接口。用户在 Single-Client 以 Eager 方式创建计算节点（创建 Tensor、执行 Operator、获取 Tensor 值等），节点被序列化为 Messages 异步发送给 Controller；Controller 将其构建为**计算子图**（每次构图并非完整计算图，而是增量子图），并通过图引擎执行。由此，DTorch 既保留 Eager 接口的易用性，又获得基于子图的全局优化能力，兼备两种模式的优点。

## C++ VS Python

Eager Graph 架构下，每个 Operator 的执行都需经过构图与调度环节，单算子调度开销直接决定框架的性能上限。作为对比，PyTorch 单个算子的调度开销为 5.94us，而 DTorch 当前单个算子的调度开销为 8.14us（仍存在优化空间）；若核心引擎基于 Python 实现，解释执行带来的额外开销将使调度开销进一步放大。

此外，Python GIL 的问题短期无法解决：多 GPU 线程共享同一进程的 GIL 会限制并行度，而基于多进程实现绕开 GIL 又会引入进程间通讯与上下文切换的开销，CPU 开销更大。

开发成本方面，PyTorch 提供 C++ 算子库 LibTorch，DTorch 以 LibTorch 作为计算后端（见 [LibTorch 后端](#libtorch-后端)），无需自行实现全部算子，相比直接使用 Python 版 PyTorch，开发成本可以接受。因此，DTorch 以 C++ 实现核心引擎，Python 仅用于框架层与用户接口；仅在必要场景（如调用仅提供 Python 接口的第三方加速库）下，才通过 [Python Kernel](#python-kernel) 将 Python 引入计算路径。

## 异步计算
Single-Client、Single-Controller 和 Multi-Worker 构成三级异步执行流水线，大幅降低系统调度开销。Client 无需等待 Controller，Controller 无需等待 Worker — 仅当用户显式获取 Tensor 值时才触发同步等待。异步计算是 DTorch 的核心特性，其使 Controller 得以提前获得计算子图，为 DTorch 带来一系列性能优化机会。

## 图改写

由于 Single-Client 与 Single-Controller 之间的异步，Controller 可提前获得计算子图，从而在子图执行前进行优化改写，包括：
- 计算与通讯重叠
- 算子融合（Fused Operator）
- 冗余节点消除
- 临时 Tensor 的显存复用
- 接入深度学习编译器进行 JIT 代码生成

## LibTorch 后端

PyTorch 提供了 C++ 算子库 **LibTorch**。DTorch 以 LibTorch 作为计算后端，大多数 Kernel 通过 `torch_kernel.cc` 调用 LibTorch API 完成实际计算。这极大地降低了算子开发成本，同时 DTorch 与 PyTorch 调用同一算子库，天然降低了精度对齐的成本。

## 并行模型

DTorch 支持三级并发，充分发挥硬件并行潜能：

**1. Graph 间并行**：可创建多个 Graph 实例，不同 Graph 在不同线程上独立执行。用户需要并发时仅需创建多个 Graph 实例，无需手动管理线程。

**2. Operator 间并行**：同一 Graph 内，不同 Operator 可分配到不同的 Stream 上并发执行。在计算图执行前，DTorch 会根据配置为每个 Operator 分配最优的 Stream（通过 `OperatorAssignInfo` 记录，见 `dtorch/core/operators/operator_assign_info.h`）。基于此可实现多线程并发计算、数据传输与计算重叠、多 GPU 并发等能力。

**3. Operator 内并行**：单个 Operator 内部也支持多种并行：
- 将计算分发到不同 Stream 上完成（即 Distributed Tensor 计算）
- CPU 上使用线程池并行化 for loop
- 利用计算设备的 SIMD、SIMT、MIMD 及数据传输与计算重叠能力

## 异步获取 Tensor 值

在三级异步执行流水线中，仅当用户获取 Tensor 的值时才触发同步等待。为避免该等待阻塞主线程，DTorch 提供 TensorFuture 机制（`dtorch/api/cpp/tensor_future.h`）：`tensor.to_torch_async()` 立即返回 TensorFuture，用户可在此期间继续创建其他计算节点，在任意时刻再获取结果。TensorFuture 支持以下使用方式：

- `get()`：阻塞等待并返回 Tensor 值
- `wait()` / `wait_for(timeout_ms)`：阻塞等待，可设置超时
- `is_ready()`：非阻塞检查值是否就绪
- `await`：TensorFuture 实现 `__await__`，可在 asyncio 协程中直接 `await future` 获取 Tensor 值，等待期间以轮询方式让出控制权，不阻塞事件循环

异步获取 Tensor 值使 Client 线程无需阻塞等待，期间可继续创建计算节点或处理其他任务，从而降低 CPU 的等待开销（overhead）。

## Python Kernel

DTorch 的 Kernel 默认通过 LibTorch C++ 算子完成计算。但部分第三方加速库仅提供 Python 接口（如 Sage Attention），为此 DTorch 支持 **Python Kernel**：在 C++ Kernel 的 `Compute()` 执行路径中调用 Python 代码，使 Operator 可按参数选择 LibTorch 原生路径或 Python 路径（如 `SdpaOp::Compute()` 根据配置调用 Sage Attention）。

Python Kernel 的每次调用遵循统一的五步模式：获取 GIL（`GilScopedAcquire`）→ 保护 CUDA Stream（`PythonCodeCudaStreamGuard`，将 Python 侧 Stream 对齐到 C++ KernelStream 的当前流）→ 导入 Python 模块 → 调用 Python 函数 → 通过 `NanobindUtil` 完成 Tensor 与 nanobind object 之间的零拷贝类型转换。由于 Python GIL 是全局互斥锁，多 GPU 线程共享同一进程的 GIL 会限制 Python Kernel 的并行度；`PerDeviceProcessNodeRunner` 通过进程隔离解决该问题：每张 GPU 运行在独立子进程中，拥有独立的 Python 解释器与 GIL，进程间通过 ZMQ IPC 通信。详见 [Python Kernel 文档](eager_graph_architecture/python_kernel.md)。
