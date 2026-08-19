# Kernel Runtime

The Operators in a LogicalGraph carry metadata only. At execution time, the framework converts each Operator into several **Kernels** (the minimal execution units), runs them on the **KernelStreams** (asynchronous execution streams) of the corresponding devices, with **Blobs** (holding `torch::Tensor`s) carrying the data reads and writes. The complete chain:

```
Graph construction: infer metadata + determine which devices the operator executes on
        │
        │  create one Kernel per (device, stream type), prepare input/output Blobs
        ▼
        Kernel ──► pushed into the KernelStream's queue, returns immediately (async)
        │
        ▼
Stream thread: loops to take Kernels → execute → release immediately (reducing peak memory)
```

## 1. Blob — the physical container of tensor data

`Blob` holds a `torch::Tensor` and sits between the Operand (metadata) and the Kernel (computation), linking the logical data node to physical memory. A distributed tensor has one Blob on each device, indexed by the global device ID; a Kernel only accesses the Blob on its own device, never reading or writing across devices.

## 2. Kernel — the minimal execution unit

`Kernel` holds its parent Operator, the input/output Blobs of its device, and the stream it belongs to. `Run()` completes the following in order: take the `torch::Tensor`s from the input Blobs → call `Compute()` (delegating to LibTorch operators by default) → write the results back to the output Blobs. Some Kernels have specialized implementations, and some can call Python code inside the Kernel (see [Python Kernel](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/python_kernel/)).

## 3. KernelStream — the asynchronous execution stream

`KernelStream` encapsulates one execution stream on a device (a CPU thread or a CUDA Stream), executing the Kernel queue in order on a **dedicated thread**. Streams come in two kinds:

| Type | Purpose |
|---|---|
| `kCompute` | regular operator computation (matmul, relu, conv2d, etc.) |
| `kCommunicate` | collective communication (all-reduce, all-gather, etc.) |

One device can have both kinds of streams simultaneously, using CUDA Stream concurrency to achieve **computation and communication overlap**.

Execution is asynchronous: a Kernel is pushed into the stream's queue and the call returns immediately; the stream thread loops to take Kernels, execute them, and release them right after execution. When synchronization is needed (such as retrieving a value), it waits for all existing Kernels in the stream to complete.

## 4. One Operator → multiple Kernels

An Operator must execute on **every device** involved in its input/output Operands, so it is instantiated as multiple Kernels: at graph construction time the framework determines the set of devices the operator must execute on from the DeviceMesh; at execution time one Kernel is created per device, each Kernel handling only its own device's computation.

```
Operator (DeviceMesh = {GPU:0, GPU:1})
   │
   ├──► Kernel(GPU:0) ──► KernelStream(GPU:0)    ← accesses only GPU 0's data
   └──► Kernel(GPU:1) ──► KernelStream(GPU:1)    ← accesses only GPU 1's data
```

## 5. Source file index

| File | Description |
|---|---|
| `dtorch/core/blob.h` `.cc` | Blob — the physical container of tensor data |
| `dtorch/core/kernel/kernel.h` `.cc` | Kernel — the minimal execution unit |
| `dtorch/core/kernel_stream/kernel_stream.h` `.cc` | KernelStream — the asynchronous execution stream |
| `dtorch/core/kernel_stream/cpu_kernel_stream.h` / `cuda_kernel_stream.h` `.cc` | the CPU / CUDA stream implementations |
| `dtorch/core/kernel_stream/kernel_stream_manager.h` `.cc` | the stream manager |
| `dtorch/core/operators/operator_assign_info.h` | the operator-to-stream assignment info |
| `dtorch/core/runner/node_runner_base.h` / `naive_runner.cc` | Runner — creates and dispatches Kernels |
