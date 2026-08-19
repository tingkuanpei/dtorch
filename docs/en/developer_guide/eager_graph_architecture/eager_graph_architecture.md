# Eager Graph Architecture

DTorch's **Eager Graph Architecture** is the core engine implementing the [Single-Client Single-Controller Multi-Worker](../single_controller.md) asynchronous distributed execution model, and is also a complete deep learning computation framework runtime. It combines the strengths of **Eager Mode** (PyTorch style, simple interface) and **Graph Mode** (TensorFlow v1 style, globally optimizable) — exposing an Eager interface externally, while optimizing and executing computation subgraphs internally.

## Design Motivation

A deep learning program usually consists of a series of Tensors and Operators. Information such as the data types and shapes of the input/output Tensors can be determined before the Operators execute; only in very rare cases (such as `nonzero`, `.item()`) is branching on the **value** of an output Tensor needed. Based on this property, DTorch can build the computation graph asynchronously on the Client side using metadata alone, without waiting for the actual computation to complete.

The design goals of the Eager Graph Architecture:

- **Eager interface**: the user writes code line by line in imperative style in Python (creating Tensors, calling Operators), with an experience identical to PyTorch's
- **Graph execution**: the Controller builds the operator sequence sent by the Client into incremental subgraphs, and optimizes them before execution
- **Asynchronous pipeline**: Client → Controller → Worker three-level asynchronous execution, with synchronous waiting only when retrieving a Tensor's value

---

## Four-Layer Architecture

The Eager Graph Architecture uses a layered design, forming a complete technology stack from the logical representation down to the low-level communication:

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

| Layer | Documentation | Responsibility |
|---|---|---|
| **Layer 1: Graph Representation** | [logical_graph_representation.md](logical_graph_representation.md) | builds a DAG from Operands (data nodes) and Operators (computation nodes), expressing the metadata of the computation logic |
| **Layer 2: Kernel Runtime** | [kernel_runtime.md](kernel_runtime.md) | converts Operators into executable Kernels, manages physical memory through Blobs, executes asynchronously on KernelStreams |
| **Layer 3: GraphExecutor** | [graph_executor.md](graph_executor.md) + [graph_executor_in_cluster.md](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/graph_executor_in_cluster/) | message-driven execution pipeline: GraphConstructor → EagerGraphExecutor → NodeRunnerBase (concrete subclasses), supporting single-machine multi-thread and single-machine multi-process modes |
| **Layer 4: Collective Communication** | [tensor_communicate.md](https://tingkuanpei.github.io/dtorch/cn/developer_guide/communicate/tensor_communicate/) | provides ThreadGroup (collective communication) and TensorStore (cross-thread tensor exchange), supporting the Placement transformations of distributed DTensors |
