# Single-Client Single-Controller Multi-Worker

Prerequisite: [Single-Controller and Multi-Controller](single_and_multi_controller.md) (an introduction to the two control paradigms).

DTorch's distributed execution model consists of one Python Client, one central Controller, and multiple Workers. The Client describes compute nodes and sends them to the Controller asynchronously; the Controller uniformly builds the computation graph and schedules resources; the Workers execute Kernels in parallel. The three advance in a pipeline through asynchronous message passing, synchronously waiting only when a Tensor's value is retrieved — combining programming ease of use with distributed execution efficiency.

## 1. Architecture Overview

When using the DTorch API, the user describes compute nodes with DTensors in single-threaded Python code (creating Tensors, calling Operators), and the framework automatically completes resource management, task dispatch and communication on the distributed cluster. DTorch contains three kinds of roles: **Client**, **Controller** and **Worker**, which together form the Single-Client Single-Controller Multi-Worker asynchronous distributed execution model:

<figure markdown>
  ![Single-Client Single-Controller Multi-Worker architecture](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/client_controller_worker_en.png)
  <figcaption>Figure 1: Client, Controller and Worker execute concurrently and communicate asynchronously through Queues</figcaption>
</figure>

### 1.1 Single-Client

**Single-Client** is the Python Client thread. The user creates Tensors, calls Operators and retrieves Tensor values in the Python Client. All operations on the Python Client are abstracted as "compute nodes" and serialized as Messages sent to the Controller asynchronously.

The Python Client does **not** directly create CUDA Memory or call CUDA Kernels to execute computation. The Tensor on the Client side is merely a **Symbol** holding metadata such as Shape, DType, DeviceMesh and Placements, but **holds no data pointer**. When executing a computation such as `tensor_c = tensor_a + tensor_b`, the Python Client can directly infer the output Tensor's Shape, DeviceMesh and other information from the input Tensors' metadata, without waiting for the actual computation to complete.

> **Client and Controller execute asynchronously**: in the vast majority of cases the Python Client completes the creation of all compute nodes without waiting for the Controller, which greatly reduces the communication overhead of the distributed system. Only when the Python Client needs to branch on the **value** of a Tensor (such as `nonzero`, `.item()`, etc.) does it need to wait for the Controller to return the computation result. DTorch also provides the **Tensor Future** mechanism, allowing asynchronous retrieval of a Tensor's value to avoid blocking the main thread.

### 1.2 Single-Controller

**Single-Controller** manages all computation resources. In the distributed cluster, there is exactly one **Main-Controller** running on the main node; on each computation node there is a **Sub-Controller** responsible for communicating with the Main-Controller and directly managing that node's resources according to its instructions. Since the Sub-Controller is subordinate to the Main-Controller, Main-Controller + Sub-Controller together are collectively called the Single-Controller.

The Controller receives the compute node Messages sent by the Client, converts them into a computation graph (LogicalGraph), and based on it completes:

- Tensor lifetime management
- CUDA Kernel creation and scheduling
- CUDA Stream creation and synchronization
- Communicate Group management
- Computation graph rewriting optimizations (operator fusion, computation-communication overlap, etc.)

The Controller translates compute nodes into executable Kernels and dispatches them to the Workers for execution.

> **Controller and Worker also execute asynchronously**, synchronizing only when a Tensor's value is retrieved.

### 1.3 Multi-Worker

**Multi-Worker** executes the actual computation tasks. Each Worker is a C++ Thread; a GPU Worker additionally holds a CUDA Stream. Workers execute the Kernels sent by the Controller in order.

Key characteristics:

- **Multiple concurrent Workers**: one GPU can correspond to multiple computation Workers and communication Workers simultaneously, enabling **overlap** of computation and communication.
- **Automatic synchronization**: a Tensor can be accessed concurrently by different Workers on the same machine; the Controller automatically inserts the necessary synchronization nodes (CUDA Events) to avoid multi-thread race conditions.

## 2. The Three-Level Asynchronous Execution Pipeline

DTorch's three roles form a **three-level asynchronous execution pipeline** running on the [producer-consumer pattern](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem):

```
Client (produces Operators)  →  Controller (produces Kernels)  →  Worker (executes Kernels)
```

Each level runs asynchronously from the next; synchronous waiting is triggered only when the user explicitly retrieves a Tensor's **value**. In current deep learning training and inference applications, branching on Tensor values is infrequent; and even when it happens, a single synchronization is cheap, so the overall performance impact is acceptable.

The three-level asynchronous pipeline brings two core benefits:

1. **Low scheduling overhead**: the Client never waits for the Controller, and the Controller never waits for the Workers — the levels advance in a pipelined fashion.
2. **Room for graph rewriting optimizations**: the Controller obtains computation subgraphs ahead of time and can optimize them before execution, including:
    - computation and communication overlap
    - operator fusion (Fused Operator)
    - redundant node elimination
    - memory reuse of temporary Tensors
    - integration of deep learning compilers for JIT code generation

## 3. Single-Controller VS Multi-Controller

DTorch's distributed interface uses Single-Controller, while PyTorch's distributed interface is a Multi-Controller based on SPMD.

### 3.1 Conceptual comparison

**Single-Controller** has only one Main-Controller node, managing **all** GPU resources in the distributed cluster; users program from the **global view** of the distributed cluster. This model was first used in TensorFlow v1 and is also discussed in [Pathways](https://arxiv.org/abs/2203.12533).

**Multi-Controller** creates multiple Controller processes, each managing only one GPU, so programming happens from the **local view** of the distributed cluster. All processes execute the same code describing the computation flow and directly schedule GPU resources (the SPMD paradigm).

<figure markdown>
  ![Single-Controller vs Multi-Controller](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/single_controller_vs_multi_controller.png)
  <figcaption>Figure 2: Single-Controller (left) and Multi-Controller (right)</figcaption>
</figure>

Single-Controller and Multi-Controller are the most fundamental difference between DTorch and PyTorch. The comparison:

| Dimension | Single-Controller | Multi-Controller | Notes |
|---|---|---|---|
| Programming view | **global view** of the distributed cluster | **local view** of the distributed cluster | |
| Ease of use of the distributed interface | good | poor | |
| Resource management | managed automatically by the system | managed manually by the user | global device management, device virtualization, automatic node failure recovery |
| Task sharding | sharded automatically by the system | sharded manually by the user | communication deadlock avoidance, Auto Parallel |
| Flexibility of the distributed interface | high | extremely high | supports the MPMD paradigm |
| Scheduling overhead | low | extremely low | DTorch reduces Single-Controller's scheduling overhead through its asynchronous architecture |

### 3.2 Advantages of Single-Controller

#### 3.2.1 Ease of use

The biggest reason DTorch chose Single-Controller is **ease of use**. Single-Controller can describe the computation from the global view of the distributed cluster, which is naturally consistent with the user's way of thinking.

The Single-Controller + DTensor approach can concisely express all kinds of parallel schemes needed for distributed deep learning computation (Data Parallel, Tensor Parallel, Pipeline Parallel, MoE Parallel, ZeRO, and orchestrating RL training workflows, etc.), greatly reducing the cost of **developing, modifying, maintaining and debugging** distributed code.

#### 3.2.2 Global optimization capability

Single-Controller provides a **global view** of the computation devices (CPU, GPU) and the computation graph, enabling:

- global device management and device virtualization
- automatic node failure recovery
- automatic communication deadlock avoidance
- Auto Parallel (automatic parallel strategy search and assignment)
- JIT Compilation (`torch.compile`, etc.)

Under Multi-Controller's SPMD paradigm, these capabilities require a great deal of manual coordination or are simply unattainable.

### 3.3 Disadvantages of Single-Controller and the responses

#### 3.3.1 Scheduling overhead

Multi-Controller has Controller nodes on every machine, and scheduling its GPUs only goes through the PCIe bus with extremely low latency. Single-Controller has only one Main-Controller, and scheduling GPUs on remote machines requires **cross-machine network communication**, so the scheduling overhead is higher than Multi-Controller's.

**DTorch's response**: adopt the **asynchronous architecture** of Single-Client Single-Controller Multi-Worker. The Client, Controller and Workers execute asynchronously through the producer-consumer pattern; blocking happens if and only if the Python Client needs to branch on a Tensor's value. In typical deep learning training and inference scenarios, such synchronization needs are infrequent and each one is cheap, so the overall performance impact is acceptable.

#### 3.3.2 Flexibility

In Multi-Controller, the code executed on each process can be completely different, thus supporting the **MPMD** (Multi Program Multi Data) paradigm with extremely high flexibility. In theory, any code requiring parallel programming can be implemented with the MPMD paradigm.

**The reality**: current large models have converged on the Transformer architecture; the excessive flexibility of the MPMD paradigm brings no actual benefit, and instead its **poor ease of use** is becoming a liability. Single-Controller keeps sufficient flexibility while providing a significantly better development experience.

## 4. Source Code Implementation

DTorch implements Single-Client Single-Controller Multi-Worker with the [Eager Graph Architecture](eager_graph_architecture/eager_graph_architecture.md). However, the class names Client, Controller and Worker are not used directly; the correspondence is:

| Role | Counterpart in the code |
|---|---|
| Single-Client | the Python thread where the user builds compute nodes. |
| Single-Controller | [class EagerGraphExecutor](https://github.com/tingkuanpei/dtorch/blob/main/dtorch/core/graph/eager_graph_executor.h) |
| Multi-Worker | [class KernelStream](https://github.com/tingkuanpei/dtorch/blob/main/dtorch/core/kernel_stream/kernel_stream.h) |
