# 集合通信组件

DTorch 的分布式执行需要跨设备、跨线程传输张量数据。`dtorch/core/communication/` 提供三套通信机制：

- **ThreadGroup** — 集合通信原语（AllReduce、AllGather 等），Kernel 进行分布式通信的入口
- **TensorStore** — 底层跨线程 / 进程的张量交换（生产者-消费者模型）
- **TensorPromise / TensorFuture** — 异步取值机制

## 1. 架构概览

三者分工明确、层层依赖：

```
Kernel (ConvertKernel / CopyKernel / GetTensorOp / …)
  │
  ├── ThreadGroup — 集合通信原语
  │     ├── 不同 GPU ──► ProcessGroupNCCL（NCCL 库）
  │     └── 同一 GPU ──► ThreadGroupSameDeviceBackend ──► TensorStore
  │
  ├── TensorStore — 直接使用
  │     ├── 张量切分 / 聚合（ScatherTensor / GatherTensor）
  │     └── 设备间拷贝（如 CPU ↔ GPU）
  │
  └── 取值 ──► TensorPromise（Worker 写入） ──► TensorFuture（Client 等待）
```

- **ThreadGroup** 是上层集合通信 API，Kernel 靠它完成 Placement 变换（AllGather、AllToAll 等）。
- **ThreadGroupSameDeviceBackend** 用于单卡模拟分布式：NCCL 不支持同 GPU 通信，该后端把集合通信全部翻译为对 **TensorStore** 的读写。
- **TensorStore** 是底层点对点张量交换：Kernel 可直接使用它完成张量切分 / 聚合（ScatherTensor / GatherTensor）与设备间拷贝（如 CPU ↔ GPU），也作为 SameDeviceBackend 的底层后端。
- **TensorPromise / TensorFuture** 解决 Client 如何非阻塞地获取 Worker 端的张量值。

## 2. ThreadGroup — 集合通信

ThreadGroup 提供集合通信原语（AllReduce / AllGather / ReduceScatter / AllToAll / Barrier），统一通过 PyTorch 的 `c10d::Backend` 接口调用，并根据设备拓扑自动选择后端：

- **不同 GPU 之间** → `ProcessGroupNCCL`：PyTorch 标准 NCCL 后端，利用 NVLink / PCIe 高带宽传输
- **同一 GPU 上** → `ThreadGroupSameDeviceBackend`：单卡模拟分布式（TP / CP 的多个进程绑在同一块 GPU）时使用

通信组由 `ThreadGroupManager` 按设备组合创建并缓存，不同 Kernel 共享；析构时按设备组合排序依次销毁，避免 NCCL 销毁顺序不一致导致死锁。

**Placement 转换矩阵** — ConvertKernel 根据源 / 目标 Placement 选择对应的集合通信原语：

| Src \ Dest | Shard(i) | Shard(j) | Replicate | Partial |
|---|---|---|---|---|
| **Shard(i)** | — | AllToAll | AllGather | — |
| **Replicate** | ReplicateToShard | — | — | — |
| **Partial** | ReduceScatter | — | AllReduce | — |

## 3. TensorStore — 跨线程张量交换

TensorStore 是**生产者-消费者**风格的张量交换抽象：一个生产者以 `key` 发布张量，N 个消费者按 `key` 读取。每次交换遵循四步握手：

```
Producer (1)                        Consumers (N)
1. SrcSet(key, tensor, N)    ──►    3. DestGet(key)         阻塞直到张量就绪
2. SrcWaitUntilGetFinished   ◄──    4. DestFinishGet(key)   通知已读完
```

生产者阻塞到所有 N 个消费者读完，消费者阻塞到生产者写入。

> **并发安全**：生产者与消费者往往运行在不同的 CUDA Stream 上，跨 Stream 同步通过 CUDA Event 完成；GPU 张量在读取完成后调用 `recordStream`，防止显存在消费完成前被提前回收。详见 [Stream Race Condition](stream_race_condition.md)。

**三种后端** — 按线程 / 进程关系自动选择：

| 后端 | 适用场景 | 底层机制 |
|---|---|---|
| MemoryTensorStore | 同一进程内多线程 | 内存 map + mutex / condition_variable |
| FileTensorStore | 同一机器多进程 | 文件 + Boost IPC + CUDA IPC |
| NetworkTensorStore | 多机器（预留） | 网络 |

**Kernel 中的使用**：DeviceMesh 迁移（ScatherTensor 把完整张量切分分发到多设备、GatherTensor 反向聚合）、CopyKernel 的跨 DeviceMesh 拷贝，都以 TensorStore 的 SrcSet / DestGet 模式完成——源设备线程作为生产者写入，目标设备线程作为消费者读取。

## 4. TensorPromise / TensorFuture — 异步取值

Client 需要获取 Tensor 值时，不阻塞等待计算完成，而是：

1. Client 创建 **TensorPromise**，立即取回 **TensorFuture**（非阻塞）
2. Promise 随 `GetTensorOp` 加入计算图，与普通算子一样异步执行
3. Worker 执行完毕后 `SetValue` 写入张量值
4. Client 需要结果时调用 `future.Get()` 阻塞等待（也可轮询 / `await`）

后端与 TensorStore 对应（同进程 Memory / 跨进程 File），由张量类型与 `perDevicePerProcess` 选项自动选择。详细机制见 [异步获取 Tensor 值](../eager_graph_architecture/async_get_tensor.md)。

## 5. 设计要点

- **三层抽象**：TensorStore 解决"线程间传张量"，ThreadGroup 解决"分布式张量如何变换"，Promise / Future 解决"Client 如何非阻塞取值"。
- **同 / 不同设备自动选择**：同 GPU 走 TensorStore（轻量，无 NCCL 开销），不同 GPU 走 NCCL（高带宽），对上层透明。
- **异步友好**：TensorStore 以 CUDA Event 同步而非全局同步，计算与通信可在不同 Stream 上重叠执行。

## 6. 源文件索引

| 文件 | 说明 |
|---|---|
| `dtorch/core/communication/thread_group/thread_group.h` `.cc` | ThreadGroup — 集合通信 API |
| `dtorch/core/communication/thread_group/thread_group_same_device_backend.h` `.cc` | 同设备后端（基于 TensorStore） |
| `dtorch/core/communication/thread_group/thread_group_manager.h` `.cc` | ThreadGroupManager — 通信组管理 |
| `dtorch/core/communication/tensor_store/tensor_store.h` `.cc` | TensorStore 基类 |
| `dtorch/core/communication/tensor_store/memory_tensor_store.h` `.cc` | Memory 后端（同进程） |
| `dtorch/core/communication/tensor_store/file_tensor_store.h` `.cc` | File 后端（跨进程） |
| `dtorch/core/communication/promise_future/tensor_promise_future.h` `.cc` | Promise / Future 基类与工厂 |
| `dtorch/core/communication/promise_future/memory_tensor_promise_future.h` `.cc` | Promise / Future Memory 后端 |
| `dtorch/core/communication/promise_future/file_tensor_promise_future.h` `.cc` | Promise / Future File 后端 |
| `dtorch/core/operators/system/get_tensor_op.h` `.cc` | GetTensorOp — 取值系统算子 |
