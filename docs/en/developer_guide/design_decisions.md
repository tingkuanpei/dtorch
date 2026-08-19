# Design Decisions

This document describes a series of key design decisions of DTorch: why the DTensor + Single-Controller approach was chosen, how the scheduling overhead of distributed systems is addressed, why LibTorch was chosen as the backend, etc. Through this document, you can understand the intent and rationale behind DTorch's architecture design.

## DTensor

Distributed parallelism requires the same Tensor to be sharded onto multiple devices, and the devices must coordinate computation and communication. If users managed this process manually, they would need to explicitly shard parameters and manually call collective communication operators — tedious and extremely error-prone code. For this reason, DTorch introduces DTensor: users declaratively describe the sharding of a Tensor through `DeviceMesh` and `Placements`, and the framework automatically completes the sharding, gathering and communication. The same code runs unchanged on one GPU or many, solving the problem of tedious and error-prone distributed parallel code.

## Single-Controller

PyTorch's distributed interface is a Multi-Controller approach based on SPMD: each process manages only one GPU, users can only program from a local view, and ProcessGroup creation, parameter sharding and collective communication calls must all be done manually; global optimization capabilities such as node failure recovery, communication deadlock avoidance and Auto Parallel require a great deal of manual coordination or are simply unattainable under the SPMD paradigm, making the distributed interface hard to use. For this reason, DTorch chose Single-Controller: a single Controller automatically manages all GPU resources in the cluster from a global view, yielding a simpler programming model and stronger global optimization capabilities.

> PyTorch's SPMD-based Multi-Controller approach is designed around ordinary Tensors; after switching to DTensor, the Single-Controller approach has better ease of use.

## Single-Client Single-Controller Multi-Worker
Single-Controller uniformly schedules all machines through the cross-machine network; compared with Multi-Controller, where each machine manages its local GPU through the PCIe bus, cross-machine scheduling brings higher scheduling overhead. To address this overhead, DTorch adopts an asynchronous execution mechanism. Deep learning programs usually consist of a series of Tensors and Operators, and metadata such as the data types and shapes of the input/output Tensors can be determined before the Operators execute. Based on this property, DTorch overlaps the construction of compute nodes, the scheduling of the distributed system, and the computation of Kernels, forming **Single-Client Single-Controller Multi-Worker** — the asynchronous distributed execution model — thereby solving the problem of Single-Controller's high scheduling overhead.

## A computation engine unifying static graphs and dynamic graphs

Deep learning framework execution modes fall into two categories: dynamic graphs and static graphs. In dynamic graphs (Eager Mode, the PyTorch architecture), Python code directly creates Tensors and launches CUDA Kernels — the interface is simple and easy to use, but there is no room for global optimization. Static graphs (Graph Mode, the TensorFlow v1 architecture) build the complete computation graph first and execute it afterwards — globally optimizable, but the interface is less intuitive. DTorch chose to unify the two: the execution engine is implemented on top of the computation graph, while exposing an Eager interface externally. The user creates compute nodes in Eager fashion on the Single-Client (creating Tensors, executing Operators, retrieving Tensor values, etc.); the nodes are serialized as Messages and sent to the Controller asynchronously; the Controller builds them into **computation subgraphs** (each construction is not a complete computation graph, but an incremental subgraph) and executes them through the graph engine. Thus DTorch keeps the ease of use of the Eager interface while gaining subgraph-based global optimization capabilities — the strengths of both modes.

## C++ VS Python

Under the Eager Graph architecture, the execution of every Operator goes through graph construction and scheduling, so the per-operator scheduling overhead directly determines the framework's performance ceiling. By comparison, PyTorch's per-operator scheduling overhead is 5.94us, while DTorch's current per-operator scheduling overhead is 8.14us (with room for optimization); if the core engine were implemented in Python, the extra overhead of interpreted execution would further magnify the scheduling overhead.

Besides, the Python GIL problem cannot be solved in the short term: multiple GPU threads sharing one process's GIL limits parallelism, while a multi-process implementation that circumvents the GIL introduces inter-process communication and context switching overheads — even higher CPU cost.

In terms of development cost, PyTorch provides the C++ operator library LibTorch; DTorch uses LibTorch as its computation backend (see [LibTorch Backend](#libtorch-backend)), so there is no need to implement all operators from scratch, and the development cost is acceptable compared with directly using the Python version of PyTorch. Therefore, DTorch implements the core engine in C++, with Python used only for the framework layer and the user interface; only in necessary scenarios (such as calling third-party acceleration libraries that only provide Python interfaces) is Python introduced into the computation path through the [Python Kernel](#python-kernel).

## Asynchronous computation
Single-Client, Single-Controller and Multi-Worker form a three-level asynchronous execution pipeline, greatly reducing the system's scheduling overhead. The Client never waits for the Controller, and the Controller never waits for the Workers — synchronous waiting is triggered only when the user explicitly retrieves a Tensor's value. Asynchronous computation is a core feature of DTorch: it allows the Controller to obtain computation subgraphs ahead of time, bringing DTorch a series of performance optimization opportunities.

## Graph rewriting

Thanks to the asynchrony between Single-Client and Single-Controller, the Controller obtains computation subgraphs ahead of time and can optimize them before execution, including:
- computation and communication overlap
- operator fusion (Fused Operator)
- redundant node elimination
- memory reuse of temporary Tensors
- integration of deep learning compilers for JIT code generation

## LibTorch Backend

PyTorch provides the C++ operator library **LibTorch**. DTorch uses LibTorch as its computation backend; most Kernels call the LibTorch API through `torch_kernel.cc` to perform the actual computation. This greatly reduces the cost of operator development, and since DTorch and PyTorch call the same operator library, the cost of precision alignment is naturally reduced.

## Parallel model

DTorch supports three levels of concurrency, fully exploiting the hardware's parallel potential:

**1. Inter-Graph parallelism**: multiple Graph instances can be created, and different Graphs execute independently on different threads. When concurrency is needed, users simply create multiple Graph instances — no manual thread management.

**2. Inter-Operator parallelism**: within the same Graph, different Operators can be assigned to different Streams to execute concurrently. Before the computation graph executes, DTorch assigns the optimal Stream to each Operator according to the configuration (recorded through `OperatorAssignInfo`, see `dtorch/core/operators/operator_assign_info.h`). Based on this, capabilities such as multi-threaded concurrent computation, data-transfer/computation overlap, and multi-GPU concurrency can be achieved.

**3. Intra-Operator parallelism**: a single Operator also supports several kinds of parallelism internally:
- dispatching the computation to different Streams (i.e., Distributed Tensor computation)
- parallelizing for loops with a thread pool on the CPU
- exploiting the computation devices' SIMD, SIMT, MIMD and data-transfer/computation overlap capabilities

## Asynchronous Tensor value retrieval

In the three-level asynchronous execution pipeline, synchronous waiting is triggered only when the user retrieves a Tensor's value. To prevent this waiting from blocking the main thread, DTorch provides the TensorFuture mechanism (`dtorch/api/cpp/tensor_future.h`): `tensor.to_torch_async()` immediately returns a TensorFuture; the user can keep creating other compute nodes in the meantime and retrieve the result at any later time. TensorFuture supports the following usage patterns:

- `get()`: block until ready and return the Tensor value
- `wait()` / `wait_for(timeout_ms)`: block until ready, with an optional timeout
- `is_ready()`: non-blocking check of whether the value is ready
- `await`: TensorFuture implements `__await__`, so `await future` can be used directly in an asyncio coroutine to retrieve the Tensor value; while waiting it yields control by polling, without blocking the event loop

Asynchronous Tensor value retrieval lets the Client thread continue creating compute nodes or handling other tasks instead of blocking, thereby reducing the CPU's waiting overhead.

## Python Kernel

DTorch's Kernels complete their computation through LibTorch C++ operators by default. However, some third-party acceleration libraries only provide Python interfaces (such as Sage Attention); for this, DTorch supports the **Python Kernel**: Python code is called in the `Compute()` execution path of a C++ Kernel, allowing an Operator to choose between the LibTorch native path and a Python path based on its parameters (e.g., `SdpaOp::Compute()` calls Sage Attention according to the configuration).

Each invocation of a Python Kernel follows a uniform five-step pattern: acquire the GIL (`GilScopedAcquire`) → protect the CUDA Stream (`PythonCodeCudaStreamGuard`, aligning the Python-side Stream to the C++ KernelStream's current stream) → import the Python module → call the Python function → complete the zero-copy type conversion between Tensors and nanobind objects through `NanobindUtil`. Since the Python GIL is a global mutex, multiple GPU threads sharing one process's GIL limits the parallelism of Python Kernels; `PerDeviceProcessNodeRunner` solves this through process isolation: each GPU runs in an independent subprocess with its own Python interpreter and GIL, and the processes communicate through ZMQ IPC. See the [Python Kernel documentation](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/python_kernel/) for details.
