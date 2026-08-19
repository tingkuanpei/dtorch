# Document Index

## Get Started

| Document | Content |
|---|---|
| [Get Started](../get_started/get_started.md) | Build and install DTorch, then verify distributed inference of the diffusion models (SD3 / FLUX) |
| [How To Build](https://tingkuanpei.github.io/dtorch/cn/get_started/how_to_build/) | Build and development environment setup |
| [Testing Guide](https://tingkuanpei.github.io/dtorch/cn/get_started/test/) | Python unit tests, C++ unit tests (gtest), application tests (Flux/SD3) |

## User Guide

| Document | Content |
|---|---|
| [User Guide](../user_guide/user_guide.md) | Getting started with DTorch: first understand the core DTensor concepts, then learn to write distributed programs with the Python API |
| [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md) | An introduction to DTensor's DeviceMesh and Placements concepts |
| [Python API Overview](../user_guide/python_api_overview.md) | Tensor creation, operator calls, native DTensor support, Placements inference, Redistribute, single-device distributed simulation, async value retrieval |
| [Module Parallel](../user_guide/module_parallel.md) | Linear's tp_dim / tp_shard_type, the ColumnParallelLinear / RowParallelLinear subclasses, DP/TP/CP/PP usage at the Module level |
| [Llama Parallel Example](../user_guide/llama_parallel.md) | Using the Llama model as an example, showing the complete DP + TP + PP + CP implementation (layer-to-stage mapping, RoPE and CP, parallel strategy testing) |
| [Diffusion Model Inference Application](https://tingkuanpei.github.io/dtorch/cn/user_guide/applications/) | Directory structure, DTorch API adaptation changes, ExecuteConfig, distribution, Cache, quantization, operator fusion |

## Architecture and Design

| Document | Content |
|---|---|
| [Project Overview](project_overview.md) | An overview of the layered architecture and directory structure |
| [Key Concepts](key_concept.md) | Detailed explanations of the three core design concepts (Single-Client Single-Controller Multi-Worker, DTensor, Eager Graph Architecture) |
| [Design Decisions](design_decisions.md) | Key design decisions: why DTensor + Single-Controller, how the scheduling overhead is addressed, the LibTorch backend, etc. |
| [Single-Controller Architecture](single_controller.md) | The Single-Client Single-Controller Multi-Worker architecture in detail, compared with Multi-Controller |
| [Distributed Tensor](distributed_tensor.md) | DTensor's DeviceMesh and Placements mechanisms, with complete code examples |

## Eager Graph Architecture

| Document | Content |
|---|---|
| [Architecture Overview](eager_graph_architecture/eager_graph_architecture.md) | The four-layer architecture (graph representation → Kernel runtime → GraphExecutor → collective communication components) |
| [Layer 1 · Graph Representation](eager_graph_architecture/logical_graph_representation.md) | Operand / Operator / LogicalGraph |
| [Layer 2 · Kernel Runtime](eager_graph_architecture/kernel_runtime.md) | Blob / Kernel / KernelStream / OperatorAssignInfo |
| [Layer 3 · GraphExecutor](eager_graph_architecture/graph_executor.md) | Single-machine multi-thread (GraphConstructor → EagerGraphExecutor → PerDeviceThreadNodeRunner → NaiveRunner) |
| [Layer 3 · Cluster Mode](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/graph_executor_in_cluster/) | Single-machine multi-process (RemoteRunnerPublisher, RemoteRunnerInProcess, PerDeviceProcessNodeRunner, SubProcess) |
| [Layer 4 · Collective Communication](communicate/tensor_communicate_overview.md) | ThreadGroup / TensorStore (Memory/File/Network three backends) |
| [Async Tensor Value Retrieval](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/async_get_tensor/) | GetTensorOp and the Promise/Future mechanism: `to_torch_async()` / TensorFuture |
| [Python Kernel](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/python_kernel/) | Calling Python code in C++ Kernels (GIL management, CUDA Stream protection, type conversion) |
| [Serialization](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/serialization/) | Operator serialization and deserialization: the Boost.Serialization system, OperatorSerializationPack, cross-process transfer |
| [Process Heartbeat](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/process_heart_beat/) | gRPC bidirectional heartbeat, MainProcessHeartBeat / WorkerProcessHeartBeat, process failure detection and graceful shutdown |
| [ZMQ Communication](https://tingkuanpei.github.io/dtorch/cn/developer_guide/communicate/zmq/) | The PUB-SUB + PUSH-PULL dual channels, message protocol and ordering guarantees |
| [Stream Race Condition](https://tingkuanpei.github.io/dtorch/cn/developer_guide/communicate/stream_race_condition/) | CUDA Stream synchronization mechanisms and cross-stream race conditions |

## Operator System

| Document | Content |
|---|---|
| [Operator System Overview](https://tingkuanpei.github.io/dtorch/cn/developer_guide/operator/operator/) | An overview of the Operator system |
| [Operator Base Class and Derived Classes](https://tingkuanpei.github.io/dtorch/cn/developer_guide/operator/operators_class/) | The base class structure and the derived class hierarchy (standard / fused_compile / system) |
| [Template Method Pattern](https://tingkuanpei.github.io/dtorch/cn/developer_guide/operator/operator_template_method_pattern/) | The algorithm skeleton and all overridable virtual functions explained |
| [PlacementSignature](https://tingkuanpei.github.io/dtorch/cn/developer_guide/operator/placement_signature/) | The distributed Placements mapping rules (Builder API, typical implementation patterns, the matching flow) |
| [Operator Mapping Table](https://tingkuanpei.github.io/dtorch/cn/developer_guide/operator/operators_mapping/) | The three-layer mapping table: Python ↔ C++ API ↔ C++ core operators |
| [How To Add an Operator](https://tingkuanpei.github.io/dtorch/cn/developer_guide/operator/how_to_add_operator/) | The five-step flow: comments → core operator → Kernel → API → tests |
| [OperatorCost Estimation](https://tingkuanpei.github.io/dtorch/cn/developer_guide/operator/operator_cost/) | FLOPs / bandwidth estimation, the base data for roofline analysis |

## Debugging and Optimization

| Document | Content |
|---|---|
| [Debugging Output Mismatch](https://tingkuanpei.github.io/dtorch/cn/developer_guide/debug_alignment_optimization/debug_output_mismatch/) | Locating the root cause of output mismatches between DTorch and PyTorch models, from coarse to fine |
| [Precision Alignment](https://tingkuanpei.github.io/dtorch/cn/developer_guide/debug_alignment_optimization/precision_alignment/) | Known scenarios of DTorch vs PyTorch output mismatch and the workarounds |
| [Performance Optimization](https://tingkuanpei.github.io/dtorch/cn/developer_guide/debug_alignment_optimization/performance_optimization/) | Analysis and optimization of performance problems such as CUDA Kernel Launch |
