# 项目概述

DTorch 是一个基于 **Single-Controller** 和 **Distributed Tensor** 架构的分布式深度学习 API，为 PyTorch 提供高性能分布式计算能力。用户可以在不修改代码逻辑的前提下，将单卡 PyTorch 程序扩展到多卡分布式环境。

- **语言**: C++（核心引擎）、Python（框架层与用户接口）
- **计算后端**: LibTorch（PyTorch C++ 算子库）
- **通信**: NCCL（GPU 集合通信）、ZMQ（节点间消息传递）、gRPC（集群 RPC）
- **构建系统**: CMake + scikit-build + nanobind

---

## 整体架构分层

DTorch 采用分层架构，从底层基础设施到顶层 API 形成完整技术栈：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Python/C++ API Layer                           │
│           Python nn.Module / nanobind bridge / C++ Tensor API            │
├──────────────────────────────────────────────────────────────────────────┤
│                          C++ Core Engine Layer                           │
│              Operator system / LogicalGraph / Kernel engine              │
│  Blob memory mgmt / Stream scheduling / Distributed runtime / Comm libs  │
├────────────────────────┬─────────────────────────────────────────────────┤
│ C++ Common Base Layer  │            External Library Adapters            │
│ Log / assert / string  │    CUDA / NCCL / ZMQ / gRPC / Boost / Torch     │
│                        │          Sage Attention / Python utils          │
├────────────────────────┴─────────────────────────────────────────────────┤
│                       Infrastructure & Build Layer                       │
│         CMake build system / Third-party deps / Install scripts          │
└──────────────────────────────────────────────────────────────────────────┘
```

| 层 | 职责 |
|---|---|
| **Python/C++ API 层** | Python 侧：PyTorch 兼容的 nn.Module 体系、HuggingFace 集成、CLI 工具、测试；nanobind 桥接；C++ 侧：Tensor、Graph、DeviceMesh 和 functional 算子的公开 API 封装 |
| **C++ 核心引擎层** | 算子定义与注册、Kernel 执行、计算图构建与执行、Runner 执行调度、Blob 内存管理、Stream 流调度、分布式运行时、集合通信 |
| **C++ 公共基础层** | 日志系统、参数解析、调试断言、字符串处理、子进程管理、环境变量、文件系统操作 |
| **外部库适配层** | 第三方库 C++ 适配封装：CUDA 设备管理、NCCL 通信、Boost 序列化、ZMQ 通信、gRPC 通信、Torch 工具、Sage Attention 加速 |
| **基础设施与构建层** | CMake 构建系统、第三方依赖管理、安装脚本、代码格式化配置、Python 打包 |

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
│   └── zmq/              # ZMQ 消息通信（RemoteRunnerPublisher/Subscriber/Pusher/Puller）
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

## 三大核心设计理念

DTorch 围绕三大核心设计理念构建：

- **Single-Client Single-Controller Multi-Worker** — 异步分布式执行模型
- **Distributed Tensor** — 原生多维分布式张量抽象
- **Eager Graph Architecture** — 融合 Eager Mode 与 Graph Mode 优势的执行引擎

各理念的设计动机详见 [设计决策](design_decisions.md) 前几章，详细介绍见 [关键概念](key_concept.md)。
