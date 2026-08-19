# Key Concepts

DTorch uses the Single-Controller architecture, controlling a distributed GPU cluster through the inter-machine network. In comparison, the Multi-Controller architecture has multiple Controllers, each controlling its local GPU only through the PCIe bus, so Single-Controller has higher scheduling overhead than Multi-Controller — in exchange for a global view and a simpler programming model.

> See [Design Decisions](design_decisions.md) for the motivation behind DTorch's choice of the Single-Controller architecture

A deep learning program usually consists of a series of Tensors and Operators; information such as the data types and shapes of the input/output Tensors can be determined before the Operator executes its computation. Only in very rare cases (such as [torch.nonzero](https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html) and [torch.Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html)) is branching based on the value of an output Tensor needed. Based on this property, DTorch can construct the computation graph of Tensors and Operators in advance, thereby reducing scheduling and runtime overhead and improving program performance.

Based on the properties above, DTorch builds a simple yet efficient deep learning API around three core concepts:

1. **Single-Client Single-Controller Multi-Worker** — asynchronous distributed execution model
2. **Distributed Tensor** — native multi-dimensional distributed tensor abstraction
3. **Eager Graph Architecture** — an execution engine combining the strengths of Eager Mode and Graph Mode

## Single-Client Single-Controller Multi-Worker

The user describes the computation with DTensors (creating Tensors, calling Operators) in single-threaded Python (the **Client**); the framework automatically handles resource management, task dispatch and communication on the distributed cluster. Three kinds of roles are involved:

```
┌──────────┐       async messages       ┌────────────┐       kernel queue        ┌────────┐
│ Client   │ ─────────────────────────> │ Controller │ ──────────────────────-─> │ Worker │
│ (Python) │                            │ (C++)      │                           │ (C++)  │
│          │ <── sync when get value ── │            │ <─ sync when get value -─ │        │
└──────────┘                            └────────────┘                           └────────┘
```

- **Client**: the user-side Python. All operations are abstracted as "compute nodes" and sent to the Controller asynchronously; the Tensor on the Client side is just a Symbol holding metadata (Shape, DType, DeviceMesh, Placements) — it holds no data and never launches CUDA Kernels directly.
- **Controller**: the single global manager, which receives compute nodes, builds the computation graph, and handles the creation, scheduling and communication management of Tensors / Kernels / Streams.
- **Worker**: the actual executor (a C++ thread; GPU Workers additionally hold a CUDA Stream), executing the Kernels dispatched by the Controller in order.

The three levels are **fully asynchronous** — Client → Controller → Worker never wait for each other; they synchronize only when a Tensor's value is needed. This is exactly the source of DTorch's low communication overhead.

See the [Single-Controller documentation](single_controller.md) for details.

## Distributed Tensor

DTorch natively supports DTensor, expressing N-D parallelism through `DeviceMesh` and `Placement`, covering all mainstream paradigms: Data Parallel, Tensor Parallel, Context Parallel, Pipeline Parallel, Expert Parallel, ZeRO, etc.

Compared to PyTorch's SPMD style, DTensor requires no manual sharding or gathering of Tensors across devices: operators automatically infer the output distribution, and weight loading and value retrieval also complete the sharding and gathering automatically, making the code simpler and more intuitive.

See the user guide [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md) and [Distributed Tensor](distributed_tensor.md).

## Eager Graph Architecture

To efficiently implement Single-Client Single-Controller Multi-Worker, DTorch designed the **Eager Graph Architecture**, combining the strengths of two execution modes:

- **Eager Mode** (PyTorch): Python directly creates Tensors and launches Kernels; the interface is simple, but there is no room for global optimization.
- **Graph Mode** (TensorFlow v1): builds the complete computation graph first and executes it afterwards; globally optimizable, but the interface is less intuitive.

Eager Graph Architecture exposes an Eager interface externally, but executes as a Graph internally: the compute nodes produced by the Client in Eager fashion are sent to the Controller asynchronously, and the Controller builds them into **computation subgraphs** (incremental subgraphs, not a one-shot whole graph) that the Graph engine then executes. This keeps the ease of use of Eager while gaining subgraph-based global optimization capabilities (computation-communication overlap, operator fusion, memory reuse, etc.).

See the [Eager Graph Architecture documentation](eager_graph_architecture/eager_graph_architecture.md) for details.
