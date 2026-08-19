# Collective Communication Components

DTorch's distributed execution needs to transfer tensor data across devices and threads. `dtorch/core/communication/` provides three communication mechanisms:

- **ThreadGroup** — collective communication primitives (AllReduce, AllGather, etc.), the entry point for Kernels to perform distributed communication
- **TensorStore** — the low-level cross-thread / cross-process tensor exchange (producer-consumer model)
- **TensorPromise / TensorFuture** — the asynchronous value retrieval mechanism

## 1. Architecture Overview

The three have clear division of labor and layered dependencies:

```
Kernel (ConvertKernel / CopyKernel / GetTensorOp / …)
  │
  ├── ThreadGroup — collective communication primitives
  │     ├── different GPUs ──► ProcessGroupNCCL (the NCCL library)
  │     └── same GPU ──► ThreadGroupSameDeviceBackend ──► TensorStore
  │
  ├── TensorStore — used directly
  │     ├── tensor sharding / gathering (ScatherTensor / GatherTensor)
  │     └── cross-device copies (e.g., CPU ↔ GPU)
  │
  └── value retrieval ──► TensorPromise (written by Worker) ──► TensorFuture (waited on by Client)
```

- **ThreadGroup** is the upper-level collective communication API; Kernels use it to complete Placement transformations (AllGather, AllToAll, etc.).
- **ThreadGroupSameDeviceBackend** is for single-device distributed simulation: NCCL does not support same-GPU communication, so this backend translates all collective communications into reads and writes on **TensorStore**.
- **TensorStore** is the low-level point-to-point tensor exchange: Kernels can use it directly for tensor sharding / gathering (ScatherTensor / GatherTensor) and cross-device copies (e.g., CPU ↔ GPU), and it also serves as the underlying backend of SameDeviceBackend.
- **TensorPromise / TensorFuture** solves how the Client retrieves the Worker-side tensor values without blocking.

## 2. ThreadGroup — collective communication

ThreadGroup provides collective communication primitives (AllReduce / AllGather / ReduceScatter / AllToAll / Barrier), uniformly invoked through PyTorch's `c10d::Backend` interface, with the backend automatically selected by the device topology:

- **Between different GPUs** → `ProcessGroupNCCL`: PyTorch's standard NCCL backend, using NVLink / PCIe high-bandwidth transfer
- **On the same GPU** → `ThreadGroupSameDeviceBackend`: used for single-device distributed simulation (multiple TP / CP processes bound to the same GPU)

Communication groups are created and cached by `ThreadGroupManager` per device combination and shared between different Kernels; at destruction they are destroyed in a sorted order per device combination, avoiding deadlocks from inconsistent NCCL destruction orders.

**The Placement transformation matrix** — ConvertKernel selects the corresponding collective communication primitive by the source / target Placements:

| Src \ Dest | Shard(i) | Shard(j) | Replicate | Partial |
|---|---|---|---|---|
| **Shard(i)** | — | AllToAll | AllGather | — |
| **Replicate** | ReplicateToShard | — | — | — |
| **Partial** | ReduceScatter | — | AllReduce | — |

## 3. TensorStore — cross-thread tensor exchange

TensorStore is a **producer-consumer** style tensor exchange abstraction: one producer publishes a tensor under a `key`, N consumers read it by `key`. Each exchange follows a four-step handshake:

```
Producer (1)                        Consumers (N)
1. SrcSet(key, tensor, N)    ──►    3. DestGet(key)         blocks until the tensor is ready
2. SrcWaitUntilGetFinished   ◄──    4. DestFinishGet(key)   notifies the read is done
```

The producer blocks until all N consumers finish reading; consumers block until the producer writes.

> **Concurrency safety**: the producer and consumers often run on different CUDA Streams; cross-stream synchronization is done through CUDA Events; GPU tensors call `recordStream` after the reads complete, preventing the memory from being reclaimed before consumption finishes. See [Stream Race Condition](https://tingkuanpei.github.io/dtorch/cn/developer_guide/communicate/stream_race_condition/) for details.

**Three backends** — automatically selected by the thread / process relationship:

| Backend | Applicable scenario | Underlying mechanism |
|---|---|---|
| MemoryTensorStore | multiple threads within one process | in-memory map + mutex / condition_variable |
| FileTensorStore | multiple processes on one machine | files + Boost IPC + CUDA IPC |
| NetworkTensorStore | multiple machines (reserved) | network |

**Usage in Kernels**: DeviceMesh migration (ScatherTensor shards a complete tensor and distributes it to multiple devices, GatherTensor gathers in reverse), and CopyKernel's cross-DeviceMesh copies, are all completed with TensorStore's SrcSet / DestGet pattern — the source device thread writes as the producer, and the target device thread reads as the consumer.

## 4. TensorPromise / TensorFuture — asynchronous value retrieval

When the Client needs a Tensor's value, it does not block waiting for the computation to finish; instead:

1. The Client creates a **TensorPromise** and immediately gets back a **TensorFuture** (non-blocking)
2. The Promise enters the computation graph with `GetTensorOp`, executing asynchronously like an ordinary operator
3. After the Worker finishes, `SetValue` writes the tensor value
4. When the Client needs the result, it calls `future.Get()` to block and wait (it can also poll / `await`)

The backend corresponds to TensorStore's (Memory within a process / File across processes), automatically selected by the tensor type and the `perDevicePerProcess` option. See [Async Tensor Value Retrieval](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/async_get_tensor/) for the detailed mechanism.

## 5. Design points

- **Three-level abstraction**: TensorStore solves "transferring tensors between threads", ThreadGroup solves "how distributed tensors transform", Promise / Future solves "how the Client retrieves values without blocking".
- **Automatic same/different device selection**: same GPU goes through TensorStore (lightweight, no NCCL overhead), different GPUs go through NCCL (high bandwidth), transparent to the upper layers.
- **Async friendly**: TensorStore synchronizes with CUDA Events instead of global synchronization, so computation and communication can overlap on different Streams.

## 6. Source file index

| File | Description |
|---|---|
| `dtorch/core/communication/thread_group/thread_group.h` `.cc` | ThreadGroup — the collective communication API |
| `dtorch/core/communication/thread_group/thread_group_same_device_backend.h` `.cc` | the same-device backend (based on TensorStore) |
| `dtorch/core/communication/thread_group/thread_group_manager.h` `.cc` | ThreadGroupManager — communication group management |
| `dtorch/core/communication/tensor_store/tensor_store.h` `.cc` | the TensorStore base class |
| `dtorch/core/communication/tensor_store/memory_tensor_store.h` `.cc` | the Memory backend (same process) |
| `dtorch/core/communication/tensor_store/file_tensor_store.h` `.cc` | the File backend (cross-process) |
| `dtorch/core/communication/promise_future/tensor_promise_future.h` `.cc` | the Promise / Future base class and factory |
| `dtorch/core/communication/promise_future/memory_tensor_promise_future.h` `.cc` | the Promise / Future Memory backend |
| `dtorch/core/communication/promise_future/file_tensor_promise_future.h` `.cc` | the Promise / Future File backend |
| `dtorch/core/operators/system/get_tensor_op.h` `.cc` | GetTensorOp — the value retrieval system operator |
