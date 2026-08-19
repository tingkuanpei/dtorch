# DTorch Architecture Design: How Simplicity and Efficiency Are Achieved Together

DTorch is a distributed deep learning API based on the Single-Controller and Distributed Tensor architecture, aiming to let users scale a single-GPU PyTorch program to a multi-GPU distributed environment without modifying the code logic. This article introduces the design motivation behind DTorch's architecture and the three core designs that support it.

Prerequisites:

- [Single-Controller and Multi-Controller](../developer_guide/single_and_multi_controller.md) — an introduction to the two control paradigms
- [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md)

## 1. The problem: Single-Controller's scheduling overhead

Single-Controller controls a distributed GPU cluster through the cross-machine network, while in Multi-Controller each machine's Controller controls its local GPU through the PCIe bus alone. Therefore, Single-Controller's scheduling overhead is higher than Multi-Controller's. What this overhead buys is the global view of the entire cluster and a simpler programming model; how to amortize it is the core problem DTorch's architecture design must solve.

The properties of deep learning programs happen to provide the room for amortization: a program usually consists of a series of Tensors and Operators, and metadata such as the data types and shapes of the input/output Tensors can be determined before the Operators execute — only in very rare cases (such as [torch.nonzero](https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html) and [torch.Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html)) does the code branch on the value of an output Tensor. Therefore, the computation graph can be constructed ahead of time, letting the graph-construction time overlap with the distributed system's scheduling and the Kernels' computation — thereby reducing the system's scheduling and runtime overhead and improving program performance.

It is exactly around this insight that DTorch builds a simple yet efficient deep learning API, whose architecture is supported by three designs:

1. **Single-Client Single-Controller Multi-Worker** — the asynchronous distributed execution model
2. **Distributed Tensor** — the native multi-dimensional distributed tensor abstraction
3. **Eager Graph Architecture** — an execution engine combining the strengths of Eager Mode and Graph Mode

The three are unfolded one by one below.

## 2. Single-Client Single-Controller Multi-Worker

When using the DTorch API, the user only needs to describe the computation with DTensors in single-threaded Python code (creating Tensors, calling Operators), and the framework automatically completes resource management, task dispatch and communication on the distributed cluster. DTorch has three kinds of roles: Client, Controller and Worker, forming the Single-Client Single-Controller Multi-Worker architecture, as shown below.

<figure markdown>
  ![Single-Client Single-Controller Multi-Worker architecture](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/client_controller_worker_en.png)
  <figcaption>Figure 1: Client, Controller and Worker execute concurrently and communicate asynchronously through Queues</figcaption>
</figure>

**Single-Client** is the Python Client. The user creates Tensors, calls Operators and retrieves Tensor values in the Python Client; these operations are all abstracted as "compute nodes", serialized into messages (Messages) and sent to the Controller asynchronously. The Python Client never directly creates CUDA Memory, nor directly launches CUDA Kernels.

The Tensor on the Client side is just a Symbol, holding metadata such as Shape, DType, DeviceMesh and Placements, without an actual data pointer. When performing a computation on a Tensor (such as `tensor_c = tensor_a + tensor_b`), the Client Thread directly infers the output Tensor's Shape, DeviceMesh and other information from the input Tensors' Shape, DeviceMesh, etc.

**The Client and the Controller execute asynchronously: in the vast majority of cases, the Python Client completes the creation of all compute nodes without waiting for the Controller, which greatly reduces the distributed system's scheduling overhead.** Only when subsequent code depends on a Tensor's value (such as branching on the value, or retrieving and printing a Tensor's value) does the Client need to wait for the Controller to finish computing and return the result. To reduce such blocking, DTorch also provides the TensorFuture mechanism to retrieve a Tensor's value asynchronously.

**Single-Controller** manages all computation resources: it receives the messages sent by the Client and organizes the compute nodes into a computation graph; then, based on the graph, it completes the creation and release of Tensors, the creation and scheduling of CUDA Kernels, the creation and synchronization of CUDA Streams, the management of Communicate Groups, and other operations. In the distributed cluster there is exactly one Main-Controller, running on the main node; each machine also has a Sub-Controller, responsible for communicating with the Main-Controller and directly managing the local resources according to its instructions. The Controller translates the compute nodes into executable Kernels and sends them to the Workers for execution. **The Controller and the Workers also execute asynchronously, synchronizing only when retrieving a Tensor's value.**

**Multi-Worker** executes the computation tasks. Each Worker is a C++ Thread, and a GPU Worker also holds a CUDA Stream. Workers execute the Kernels dispatched by the Controller in order. One GPU can correspond to multiple computation Workers and communication Workers, achieving the overlap of computation and communication. The same Tensor may be read and written by multiple Workers on the same machine simultaneously; the Controller automatically inserts the necessary synchronization nodes (CUDA Events) to avoid multi-thread race conditions.

## 3. Distributed Tensor

**DTorch natively supports DTensor, and has built a complete, easy-to-use distributed API on top of DTensor.** PyTorch also supports DTensor, but it is still in ["alpha state and under development"](https://docs.pytorch.org/docs/stable/distributed.tensor.html). DTensor expresses N-D parallelism through DeviceMesh and Placements (Replicate, Shard, Partial), supporting all current forms of parallelism, including Data Parallel, Tensor Parallel, Context Parallel, Pipeline Parallel, Expert Parallel and ZeRO.

Compared to the ordinary Tensors of PyTorch's SPMD style, DTensor requires no manual sharding or gathering of Tensors across devices — the code is concise and intuitive. For example, implementing Tensor Parallel with PyTorch SPMD Tensors requires explicitly sharding the Linear weights; when results need to be gathered during computation, collective communication operators such as all-gather must also be called manually — tedious and extremely error-prone code.

### 3.1 Operators natively support DTensor

DTorch supports creating DTensors directly (providing Tensor creation interfaces such as `randn`, `ones`, `empty`), and all Operators natively support DTensor — just like single-machine Tensors, directly call the Operator you need. Every Operator has a set of `PlacementSignature` rules describing the correspondence between input/output Placements. Based on the `PlacementSignature` and other rules, the framework automatically infers the output Tensor's DeviceMesh and Placements from the input Tensors' DeviceMesh and Placements.

When the input Tensors' DeviceMesh or Placements cannot satisfy the operator's `PlacementSignature`, a `tensor.redistribute()` operation needs to be inserted to convert them into the distribution the operator requires. Because this operation is usually expensive, DTorch does not insert it automatically by default — it raises an error asking the user to modify the code. For ease of use, a few operators insert `redistribute()` automatically (such operators are called out in the documentation), for example: adding a Shard Tensor and a Replicate Tensor, calling `tensor.sum()` on a Shard Tensor, and scaled dot-product attention supporting Ulysses and Ring Context Parallel.

### 3.2 Communication as operators

DTorch changes a Tensor's DeviceMesh and Placements through `tensor.redistribute()`. Users do not need to explicitly create and manage a [ProcessGroup](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group), nor manually call collective communication operators such as [all_reduce](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) — the DeviceMesh in DTorch merely represents a set of Device IDs, and the framework creates, manages and invokes the corresponding ProcessGroup at the bottom.

In PyTorch, collective communication must explicitly specify a ProcessGroup and implicitly runs on the CUDA Stream bound to it; in DTorch, `redistribute()` is just an ordinary operator, whose bottom layer can choose the most efficient implementation according to the computation graph and network topology, and achieve the overlap of computation and communication — better performance while keeping the interface easy to use. DTorch also supports uneven sharding of DTensors: since some of NCCL's collective communication primitives require the input Tensors' shapes to be identical on all ranks, `redistribute()` automatically pads or removes padding as needed.

### 3.3 Freely composable parallel schemes

These forms of parallelism are implemented differently: Data Parallel and Pipeline Parallel only need the correct DeviceMesh and Placements configured; Tensor Parallel provides parallel Modules such as `ColumnParallelLinear`, `RowParallelLinear` and `EmbeddingWithReplicateOutput` for users to call; Context Parallel is enabled through the `ulysess_cp` / `ring_cp` named dimensions in the DeviceMesh — when `scaled_dot_product_attention` detects the corresponding dimension, it automatically switches to the Ulysses or Ring CP implementation. See [Module Parallel](../user_guide/module_parallel.md) for the concrete usage of each form of parallelism at the Module level.

PyTorch, constrained by compatibility with existing Module code, can only implement parallelism by adding hooks to Modules via [parallelize_module](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module); DTorch uses these parallel interfaces directly — simple and intuitive, with no need for users to understand complex hook invocation logic. All of DTorch's forms of parallelism can be freely combined: the same code can run on a single GPU, or with different forms of parallelism enabled in combination; see [Llama Parallel Example](../user_guide/llama_parallel.md) for the complete DP + TP + PP + CP implementation on the Llama model.

Thanks to the Single-Controller and DTensor architecture, after enabling Tensor Parallel, Pipeline Parallel and Expert Parallel, loading a model's state_dict requires no explicit sharding and reading of the corresponding model parameters (unlike the sharding of Parameters by TP and PP in Megatron-LM) — all sharding operations are implicitly completed by the framework according to the Tensors' DeviceMesh and Placements.

## 4. Eager Graph Architecture

To efficiently implement the Single-Client Single-Controller Multi-Worker paradigm, DTorch designed a brand-new execution architecture: **Eager Graph Architecture**, combining the strengths of Eager Mode and Graph Mode. Eager Mode is the architecture used by PyTorch: Python code directly creates Tensors on the GPU and launches CUDA Kernels — the interface is simple and easy to use. Graph Mode is the architecture used by TensorFlow v1: build the computation graph first and execute the computation afterwards — global optimization can be performed based on the graph, with better performance, but the interface is less intuitive.

Eager Graph Architecture is a Graph-based execution engine that exposes an Eager interface externally. The user creates compute nodes in Eager fashion on the Single-Client (creating Tensors, executing Operators, retrieving Tensor values, etc.); the compute nodes are serialized into messages and sent to the Controller asynchronously; the Controller creates computation subgraphs from the received messages (not a complete computation graph, but incremental subgraphs) and hands them to the Graph engine for execution. Thus DTorch keeps the ease of use of the Eager interface while gaining subgraph-based global optimization capabilities.


<figure markdown>
  ![Eager Graph architecture diagram](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/eager_graph_architecture.png)
  <figcaption>Figure 2: The Eager Graph Architecture diagram</figcaption>
</figure>

### 4.1 Graph representation

DTorch represents the computation graph with three abstractions: Operand, Operator and LogicalGraph, containing only the graph's metadata (no data pointers, no CUDA Kernel execution code, etc.). On the Single-Client, an Operator can infer the output Operands' metadata from the input Operands' metadata (DataKind, Device and Shape), so the computation graph can be built asynchronously.

### 4.2 Graph execution

Following the paradigm of CUDA Streams, DTorch proposed an execution engine composed of Blob, Kernel and Stream:

- **Blob** represents a tensor, holding allocated memory (CPU Memory or CUDA Memory).
- **Kernel** performs computation on input Blobs and writes the results to output Blobs; inside it is CPU code or a CUDA Kernel — it is the unit that truly executes the computation.
- **Stream** is the unified abstraction of the execution resources provided by the compute chip, such as CPU threads and CUDA Streams. Kernels must execute on Streams; computation on different Streams is mutually independent and executes concurrently; Streams support synchronization through Events.

### 4.3 Asynchronous computation and graph rewriting

Eager Graph Architecture has two core advantages: **asynchronous computation** and **graph rewriting**.

Single-Client, Single-Controller and Multi-Worker form an asynchronous execution engine, reducing the system's scheduling overhead. At the same time, thanks to the asynchrony between Single-Client and Single-Controller, the Controller can obtain computation subgraphs ahead of time and rewrite them before execution, achieving optimizations such as computation-communication overlap, operator fusion, redundant node elimination and memory reuse of temporary Tensors; going further, deep learning compilers can be integrated for JIT code generation.

### 4.4 The three-level concurrency model

DTorch supports parallelism at three levels, fully exploiting the hardware's concurrency potential: **inter-Graph**, **inter-Operator**, **intra-Operator**.

**Inter-Graph parallelism**: multiple Graphs can be created in DTorch, and different Graphs execute on different threads. Therefore, when concurrency is needed, users simply create multiple Graph instances — no need to create and manage threads manually.

**Inter-Operator parallelism**: within the same Graph, assigning different Operators to different Streams achieves parallelism between Operators. Before the computation graph executes, DTorch assigns the optimal Stream to each Operator according to the configuration. Based on this, capabilities such as multi-threaded concurrent computation, data-transfer/computation overlap and multi-GPU concurrency can be achieved.

**Intra-Operator parallelism**: a single Operator supports several kinds of parallelism internally:

- dispatching one Operator's computation to different Streams (i.e., Distributed Tensor computation);
- parallelizing for loops with a thread pool on the CPU;
- exploiting the SIMD, SIMT, MIMD and data-transfer/computation overlap capabilities provided by the compute devices.

### 4.5 The LibTorch backend

PyTorch provides a C++ operator library called [LibTorch](https://pytorch.org/cppdocs/). DTorch uses LibTorch as its computation backend, greatly reducing the development cost of operators; at the same time, since DTorch and PyTorch call the same operator library, the cost of operator precision alignment is naturally reduced.

## 5. Summary

Single-Controller trades the scheduling overhead of the cross-machine network for a global view and a simpler programming model; and the property of deep learning programs — "metadata can be determined ahead of time" — makes this cost amortizable through asynchronous execution. Around this point, DTorch's architecture consists of three designs that complement each other:

| Design | Problem solved | Core mechanism |
|---|---|---|
| Single-Client Single-Controller Multi-Worker | Single-Controller's high scheduling overhead | Client → Controller → Worker three-level fully asynchronous pipeline, synchronizing only when retrieving a Tensor's value |
| Distributed Tensor | distributed code is tedious and error-prone | DeviceMesh + Placements declaratively describe sharding; sharding, gathering and communication are implicitly completed by the framework |
| Eager Graph Architecture | Eager is hard to optimize globally, Graph's interface is unintuitive | exposes an Eager interface externally, executes incremental subgraphs internally, supports asynchronous computation and graph rewriting |

The combined effect of the three: users describe the computation in the way single-GPU PyTorch is written, and scale to a multi-GPU distributed environment without modifying the code logic. Behind the scenes the framework completes scheduling, sharding and communication, and builds optimizations such as computation-communication overlap, operator fusion and memory reuse on top of the computation subgraphs obtained ahead of time — simplicity and efficiency are thereby achieved together.

DTorch's code and documentation are both open source on [GitHub](https://github.com/tingkuanpei/dtorch); attention and participation are welcome.

## Further Reading

- [Key Concepts](../developer_guide/key_concept.md) — the three core designs explained
- [Design Decisions](../developer_guide/design_decisions.md) — the motivations and approaches of the key design decisions
- [Single-Controller Architecture](../developer_guide/single_controller.md) / [Distributed Tensor](../developer_guide/distributed_tensor.md) / [Eager Graph Architecture](../developer_guide/eager_graph_architecture/eager_graph_architecture.md)
- [User Guide](../user_guide/user_guide.md) — writing distributed programs with the Python API
