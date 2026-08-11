# DTorch 架构文档

## 项目概述

DTorch 是一个基于 **Single-Controller** 和 **Distributed Tensor** 架构的分布式深度学习 API，为 PyTorch 提供高性能分布式计算能力。用户可以在不修改代码逻辑的前提下，将单卡 PyTorch 程序扩展到多卡分布式环境。

- **语言**: C++（核心引擎）、Python（框架层与用户接口）
- **计算后端**: LibTorch（PyTorch C++ 算子库）
- **通信**: NCCL（GPU 集合通信）、ZMQ（节点间消息传递）、gRPC（集群 RPC）
- **构建系统**: CMake + scikit-build + nanobind

---

## 整体架构分层

DTorch 采用分层架构，从底层基础设施到顶层 API 形成完整技术栈：

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Python/C++ API 层                              │
│  Python nn.Module / HuggingFace / nanobind 桥接 / C++ Tensor API     │
├──────────────────────────────────────────────────────────────────────┤
│                       C++ 核心引擎层                                 │
│  Operator 算子体系 / Kernel 执行引擎 / LogicalGraph 计算图           │
│  Blob 内存管理 / Stream 流调度 / 分布式运行时 / 通信库               │
├──────────────────────┬───────────────────────────────────────────────┤
│  C++ 公共基础层      │  外部库适配层                                 │
│  日志/断言/字符串    │  CUDA / NCCL / ZMQ / gRPC / Boost / Torch     │
│                      │  Sage Attention / Python 工具                 │
├──────────────────────┴───────────────────────────────────────────────┤
│                      基础设施与构建层                                │
│  CMake 构建系统 / 第三方依赖管理 / 安装脚本 / CI                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 各层职责

| 层 | 职责 |
|---|---|
| **Python/C++ API 层** | Python 侧：PyTorch 兼容的 nn.Module 体系、HuggingFace 集成、CLI 工具、测试；nanobind 桥接；C++ 侧：Tensor、Graph、DeviceMesh 和 functional 算子的公开 API 封装 |
| **C++ 核心引擎层** | 算子定义与注册、Kernel 执行、计算图构建与执行、Runner 执行调度、Blob 内存管理、Stream 流调度、分布式运行时、集合通信 |
| **C++ 公共基础层** | 日志系统、参数解析、调试断言、字符串处理、子进程管理、环境变量、文件系统操作 |
| **外部库适配层** | 第三方库 C++ 适配封装：CUDA 设备管理、NCCL 通信、Boost 序列化、ZMQ 通信、gRPC 通信、Torch 工具、Sage Attention 加速 |
| **基础设施与构建层** | CMake 构建系统、第三方依赖管理、安装脚本、代码格式化配置、Python 打包 |

---

## 三大核心设计理念

DTorch 围绕三大核心设计理念构建。详见 [设计理念文档](design_concept.md)。

### 1. Single-Client Single-Controller Multi-Worker

异步分布式执行模型，三类角色协作：

```
┌──────────────┐      异步消息       ┌─────────────────┐       Kernel Queue      ┌─────────-─┐
│  Client      │ ─────────── ────> │                  │ ──────────────────────> │           │
│  (Python)    │                   │    Controller    │                         │   Worker  │
│              │ <── 仅取值时同步 ── │                  │ <─---─ 仅取值时同步 ────  │           │
└──────────────┘                   └──────────────── ─┘                         └──────────-┘
```

- **Single-Client**: Python 进程，创建 DTensor Symbol 和 Operator，序列化为消息异步发送。Client 侧 Tensor 仅持有元信息（Shape、DType、DeviceMesh、Placements），不持有数据。
- **Single-Controller**: C++ MainNode（`dtorch/core/distributed/main_node.h`），接收消息构建 LogicalGraph，管理全部计算资源。每台机器有 Sub-Controller（`dtorch/core/distributed/worker_node.h`）管理本机资源。
- **Multi-Worker**: C++ Thread，每个 Worker 绑定一个 CUDA Stream 或 CPU 线程，按序执行 Kernel。

### 2. Distributed Tensor (DTensor)

通过 `DeviceMesh` 和 `Placements` 原生支持 N-D 并行，详见 [Distributed Tensor 文档](../user_guide/distributed_tensor.md)。

- **DeviceMesh** (`dtorch/api/cpp/distributed_spec.h`): N 维设备网格，描述集群 GPU 拓扑
- **Placements**: 三种分布策略 — `Shard(dim)`（切分）、`Replicate()`（复制）、`Partial()`（部分聚合）
- 统一表达 Data Parallel、Tensor Parallel、Context Parallel、Pipeline Parallel、Expert Parallel、ZeRO 等所有并行策略

### 3. Eager Graph Architecture

融合 Eager Mode（简洁接口）和 Graph Mode（全局优化）的执行引擎，采用四层分层设计：

```
┌──────────────────────────────────────────────────────────────────────┐
│                 Eager Graph 引擎 (Layer 3)                            │
│  GraphConstructor → EagerGraphExecutor → RunnerBase (4 种派生 Runner) │
│  消息队列驱动，异步消费 Operator，管理计算图生命周期                    │
│                                                                      │
│  单机多线程：PerDeviceThreadNodeRunner → NaiveRunner (kMemory store)  │
│  单机多进程：PerDeviceProcessNodeRunner → RemoteRunnerPublisher/      │
│              RemoteRunnerInProcessClientAndServer → ZMQ → 子进程       │
├──────────────────────────────────────────────────────────────────────┤
│     计算图表示 (Layer 1)              Kernel 运行时 (Layer 2)          │
│  Operand / Operator / LogicalGraph    Blob / Kernel / KernelStream   │
│  DAG 拓扑结构，元信息推导              物理内存容器，异步执行流          │
├──────────────────────────────────────────────────────────────────────┤
│                      集合通讯组件 (Layer 4)                            │
│  ThreadGroup (集合通信原语)  /  TensorStore (跨线程张量交换)            │
│  AllReduce, AllGather, ReduceScatter ... Memory / File / Network       │
└──────────────────────────────────────────────────────────────────────┘
```

- **Layer 1 — 计算图表示**: Operand（张量元信息节点）、Operator（计算节点）、LogicalGraph（DAG 容器）。仅含元信息，不含数据指针或 CUDA Kernel。
- **Layer 2 — Kernel 运行时**: Blob（torch::Tensor 物理容器）、Kernel（最小执行单元）、KernelStream（CPU 线程/CUDA Stream 封装）。Operator → Kernel 映射，异步执行。
- **Layer 3 — Eager Graph 引擎**: GraphConstructor（Python API 桥梁）→ EagerGraphExecutor（Controller 实现，AsyncMain 消息循环）→ RunnerBase 派生类（NaiveRunner 执行引擎）。支持单机多线程和单机多进程两种模式。
- **Layer 4 — 集合通讯组件**: ThreadGroup（AllReduce/AllGather 等集合通信原语）+ TensorStore（生产者-消费者跨线程张量交换）。支撑 DTensor 的 DeviceMesh 变换和 Placement 重分布。
- **三级异步流水线**: Client → Controller → Worker，仅取值时同步
- **图改写优化**: 计算-通信重叠、算子融合、冗余消除、显存复用

---

## 目录结构总览

```
dtorch/
├── api/
│   ├── cpp/              # C++ 公开 API（Tensor, DeviceMesh, functional）
│   └── python/           # nanobind Python 绑定
├── common/               # 公共基础库（日志、调试、字符串、文件系统）
├── core/
│   ├── blob/             # Blob 物理内存容器
│   ├── communication/    # TensorStore、ThreadGroup 通信抽象
│   ├── distributed/      # MainNode / WorkerNode 分布式节点管理
│   ├── graph/            # LogicalGraph / GraphConstructor / EagerGraphExecutor
│   ├── kernel/           # Kernel 执行引擎（TorchKernel 等）
│   ├── kernel_stream/    # Stream 流调度（Cpu/CudaKernelStream）
│   ├── operand/          # Operand 张量元信息
│   ├── operators/        # Operator 算子体系
│   │   ├── standard/     #   标准算子（LibTorch 后端）
│   │   ├── fused_compile/#   融合编译算子
│   │   ├── system/       #   系统算子
│   │   └── ...           #   算子基类、工厂、PlacementSignature
│   └── runner/           # Runner 执行层（NaiveRunner, PerDeviceThread/ProcessNodeRunner 等）
├── external/
│   ├── boost/            # Boost 序列化适配
│   ├── cuda/             # CUDA 设备/流/事件管理
│   ├── nccl/             # NCCL 通信适配
│   ├── python/           # Python 工具（类型解析等）
│   ├── rpc/              # gRPC 服务定义与实现
│   ├── sage_attn/        # Sage Attention 加速适配
│   ├── torch/            # LibTorch 工具
│   └── zmq/              # ZMQ 消息通信（RemoteRunnerPublisher/Client/Server）
└── tests/                # C++ 单元/集成测试

python/dtorch/
├── __init__.py           # 包入口，API 导出
├── nn/                   # nn.Module 体系
│   ├── module.py         #   DTorchModule 基类
│   ├── linear.py         #   Linear / 分布式变体
│   └── ...
├── applications/
│   └── huggingface/
│       ├── transformers/ #   T5 等模型
│       └── diffusers/    #   SD3 / Flux（UNet, Pipeline, Scheduler）
└── test/                 # Python 测试套件
    ├── operators/        #   算子测试（PyTorch 基准）
    ├── modules/          #   模块测试
    └── ...
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [python_api.md](../user_guide/python_api.md) | Python API 使用指南：Tensor 创建、算子调用、DTensor 原生支持、Placements 推导、Redistribute、DP/TP 示例 |
| [applications.md](../user_guide/applications.md) | 扩散模型推理应用：目录结构、DTorch API 适配修改、ExecuteConfig、分布式、Cache、量化、算子融合 |
| [design_concept.md](design_concept.md) | 三大核心设计理念详解（Single-Controller、DTensor、Eager Graph） |
| [distributed_tensor.md](../user_guide/distributed_tensor.md) | DTensor 的 DeviceMesh 与 Placements 机制，含完整代码示例 |
| [test.md](../get_started/test.md) | 测试指南：Python 单元测试、C++ 单元测试 (gtest)、应用测试（Flux/SD3） |
| [how_to_build.md](../get_started/how_to_build.md) | 构建与开发环境配置 |
| [single_controller.md](single_controller.md) | Single-Client Single-Controller Multi-Worker 架构详解，与 Multi-Controller 对比 |
| [eager_graph_architecture.md](eager_graph_architecture/eager_graph_architecture.md) | Eager Graph 架构总览：四层架构（计算图表示 → Kernel 运行时 → Eager Graph 引擎 → 集合通讯组件） |
| [eager_graph_engine.md](eager_graph_architecture/eager_graph_engine.md) | Layer 3 详解：单机多线程（GraphConstructor → EagerGraphExecutor → PerDeviceThreadNodeRunner → NaiveRunner） |
| [eager_graph_engine_in_cluster.md](eager_graph_architecture/eager_graph_engine_in_cluster.md) | Layer 3 详解：单机多进程（RemoteRunnerPublisher, RemoteRunnerInProcessClientAndServer, PerDeviceProcessNodeRunner, SubProcess） |
| [zmq.md](communicate/zmq.md) | ZMQ 通信机制：PUB-SUB + REQ-REP 双通道、消息协议与顺序保证 |
| [process_heart_beat.md](eager_graph_architecture/process_heart_beat.md) | 进程心跳机制：gRPC 双向心跳、MainProcessHeartBeat / WorkerProcessHeartBeat、进程故障检测与优雅关闭 |
| [serialization.md](eager_graph_architecture/serialization.md) | Operator 序列化与反序列化：Boost.Serialization 体系、OperatorSerializationPack、跨进程传输 |
| [logical_graph_representation.md](eager_graph_architecture/logical_graph_representation.md) | Layer 1: 计算图表示 — Operand / Operator / LogicalGraph |
| [kernel_runtime.md](eager_graph_architecture/kernel_runtime.md) | Layer 2: Kernel 运行时 — Blob / Kernel / KernelStream / OperatorAssignInfo |
| [python_kernel.md](eager_graph_architecture/python_kernel.md) | 在 C++ Kernel 中调用 Python 代码（GIL 管理、CUDA Stream 保护、类型转换） |
| [tensor_communicate.md](communicate/tensor_communicate.md) | Layer 4: 集合通讯组件 — ThreadGroup / TensorStore（Memory/File/Network 三种后端） |
| [operator/operator.md](operator/operator.md) | Operator 算子体系总览 |
| [operator/operators_class.md](operator/operators_class.md) | Operator 基类结构与派生类体系（standard / fused_compile / system） |
| [operator/operator_template_method_pattern.md](operator/operator_template_method_pattern.md) | 模板方法模式的算法骨架与所有可重写虚函数详解 |
| [operator/placement_signature.md](operator/placement_signature.md) | 分布式 Placements 映射规则（Builder API、典型实现模式、匹配流程） |
| [operator/operators_mapping.md](operator/operators_mapping.md) | Python↔C++ API↔C++ 核心算子三层映射表 |
| [operator/how_to_add_operator.md](operator/how_to_add_operator.md) | 新增算子开发指南（注释→核心算子→Kernel→API→测试五步流程） |
