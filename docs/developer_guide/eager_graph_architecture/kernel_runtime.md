# Kernel 运行时：从 Operator 到 KernelStream 的执行映射

DTorch 的 GraphExecutor 在构建 LogicalGraph 之后，需要将图中的 Operator 节点转化为可在具体设备上执行的 Kernel，并分配到对应的 KernelStream 上异步执行。本文档描述 **Blob**、**Kernel**、**KernelStream**、**KernelStreamKey** 和 **OperatorAssignInfo** 如何协作完成这一过程。

## 1. 架构概览

```
Operator::Infer()
  │
  └─→ InferOperatorAssignInfo()   ──→  OperatorAssignInfo (StreamKeySet)
                                          │
                                          ▼
Runner::CreateKernelForOperator()         ──→  为每个 KernelStreamKey 创建一个 Kernel
  │                                           │
  ├─ 创建/查找 Blob（输入/输出张量容器）        │
  ├─ KernelStreamManager::GetStream()          │
  └─ Kernel::CreateKernel(ctx)                 │
                                          │
                                          ▼
Runner::Execute()
  │
  └─→ kernel->GetStream().LaunchKernel(kernel)  ──→  KernelStream 异步队列
                                                       │
                                                       ▼
                                                KernelStream::AsyncMain()
                                                       │
                                                       └─→ Kernel::Run()
                                                             ├─ 从 Blob 读取输入 torch::Tensor
                                                             ├─ Compute() → Operator::Compute()
                                                             └─ 将输出 torch::Tensor 写回 Blob
```

**核心思想**：每个 Operator 通过其输入/输出 Operand 的 DeviceMesh 确定需要在哪些设备上执行，为每个 `(Device, StreamType)` 组合创建一个 Kernel，每个 Kernel 被分派到对应的 KernelStream 上异步执行。

## 2. Blob — 张量数据的物理容器

**源文件**: `dtorch/core/blob.h`

`Blob` 是 `torch::Tensor` 的引用计数包装器，是 Kernel 读写张量数据的**物理容器**。它位于 Operand（元信息）和 Kernel（计算）之间，将逻辑数据节点与物理内存关联起来。

### 结构

```cpp
class Blob {
    struct Impl {
        std::shared_ptr<torch::Tensor> mTensor;  // 实际的 torch::Tensor
        std::mutex mMutex;                       // 线程安全保护
    };
    std::shared_ptr<Impl> mImplPtr;

    bool IsEmpty() const;
    void SetTensor(const std::shared_ptr<torch::Tensor>& tensor);
    std::shared_ptr<torch::Tensor> GetTensor() const;
    void CreateTensor(const Shape& shape, const Device& device, DataKind dataKind);
};
```

### AllDeviceBlobs

```cpp
using AllDeviceBlobs = std::unordered_map<int64_t, Blob>;
```

对于一个分布在多个设备上的 Operand，其对应的 `AllDeviceBlobs` 是一个按**全局设备 ID** 索引的 Blob 映射。例如，一个分布在 GPU 0 和 GPU 1 上的张量：

```
AllDeviceBlobs = {
    {0, Blob(GPU:0 上的 torch::Tensor)},
    {1, Blob(GPU:1 上的 torch::Tensor)},
}
```

### Blob 在 Kernel 中的角色

每个 Kernel 的构造函数（`kernel.cc`）会从 `KernelCreateCtx` 中提取与自身 `mGlobalDevice` 相关的 Blob：

- **输入**: 对于每个输入 Operand，如果其 `DeviceMesh` 包含 `mGlobalDevice`，则从 `AllDeviceBlobs` 中取出对应的 `Blob` 存入 `mInputs`；否则存入 `std::nullopt`。
- **输出**: 同理，将对应的 `Blob` 存入 `mOutputs`。

> 这意味着每个 Kernel 只持有与自身设备相关的 Blob 引用，不会跨设备访问数据。

## 3. Kernel — 执行单元

**源文件**: `dtorch/core/kernel/kernel.h`

`Kernel` 是 DTorch 中的**最小执行单元**。一个 Operator 在 `NumKernelForThisOp()` 个 `DeviceStream` 流上执行，就对应创建 `NumKernelForThisOp()` 个 Kernel。

### KernelCreateCtx — 创建上下文

```cpp
struct KernelCreateCtx {
    std::shared_ptr<Operator> op;                     // 父算子
    std::vector<AllDeviceBlobs> inputs;               // 每个输入 Operand 的 AllDeviceBlobs
    std::vector<AllDeviceBlobs> outputs;              // 每个输出 Operand 的 AllDeviceBlobs
    KernelStream* stream;                             // 分配到哪个流
    Device globalDevice;                              // 全局设备标识
    Device localDevice;                               // 本机设备标识
    size_t numKernelForThisOp;                        // 该算子总共创建几个 Kernel
    ThreadGroupManager* threadGroupManager;           // 线程组管理器
};
```

### Kernel 核心成员

```cpp
class Kernel {
    std::shared_ptr<Operator> mOp;                    // 父算子（持有参数）
    std::vector<std::optional<Blob>> mInputs;         // 输入 Blob（按此 Kernel 设备过滤后）
    std::vector<std::optional<Blob>> mOutputs;        // 输出 Blob
    Device mGlobalDevice;                             // 此 Kernel 运行的全局设备
    Device mLocalDevice;                              // 此 Kernel 运行的本地设备
    KernelStream* mStream;                            // 所属流
    std::vector<std::shared_ptr<KernelHook>> mKernelHooks;  // 计算前后钩子
    std::shared_ptr<TensorStoreCreateInfo> mTensorStoreCreateInfo; // 跨 Kernel 通信
};
```

### 工厂方法

`Kernel::CreateKernel(ctx)` 根据 `OperatorType` 分派创建不同的 Kernel 子类：`ConvertKernel`、`CreateKernel`、`CopyKernel`、`ViewKernel`、`ReduceKernel`、`MemoryKernel` 或基础 `Kernel`。

### 执行流程：Kernel::Run()

```
Kernel::Run()
  │
  ├─ 1. KernelHook::BeforeCompute()  — 调用所有前置钩子
  ├─ 2. CUDA Stream 校验             — 确保当前 CUDA Stream 未被切换
  ├─ 3. 准备输入                     — 从 mInputs 的 Blob 中提取 torch::Tensor
  │     ├─ 验证 Shape、DataKind、Device
  │     └─ 构建 TorchTensorOptArray
  ├─ 4. Compute(inputs, outputs)     — 调用子类或 Operator::Compute()
  ├─ 5. 验证输出                     — 检查输出 Shape、DataKind、Device
  ├─ 6. 写回 Blob                    — 通过 Blob::SetTensor() 将结果写回 mOutputs
  └─ 7. KernelHook::AfterCompute()   — 调用所有后置钩子
```

默认的 `Compute()` 实现直接委托给 `mOp->Compute(torch_inputs, torch_outputs)`，即调用底层 LibTorch 算子。部分 Kernel 子类（如 `CopyKernel`、`CreateKernel`）会覆盖 `Compute()` 提供特化逻辑。

## 4. KernelStream — 异步执行流

**源文件**: `dtorch/core/kernel_stream/kernel_stream.h`

`KernelStream` 是 Cpu Thread 和 Cuda Stream 的封装，每个流绑定一个特定设备的一个专用线程，按序执行 Kernel 队列。

### 核心结构

```cpp
class KernelStream {
    Device mDevice;                    // 此流绑定的本地设备
    KernelStreamType mStreamType;      // kCompute 或 kCommunicate
    bool mIsAsync;                     // 是否异步执行（专用线程）
    std::thread mAsyncStreamThread;    // 异步执行线程
    KernelQueue mKernelQueue;          // 无锁队列（BlockingReaderWriterQueue）
    SharedInfo mSharedInfo;            // 同步状态（mutex, cv, destroy, sync 标志）
};
```

### 异步执行模型

当前所有 KernelStream 均以异步模式（`mIsAsync = true`）运行：

```
LaunchKernel(kernel)                   AsyncMain() 线程
     │                                      │
     ├─ 将 kernel 推入 mKernelQueue  ──→   ├─ InitInAsyncThread()
     │                                      │   ├─ CpuKernelStream: 设置线程名称
     │                                      │   └─ CudaKernelStream: 获取 CUDA Stream
     │                                      │       ├─ cudaSetDevice(deviceId)
     │                                      │       ├─ getStreamFromPool()
     │                                      │       └─ setCurrentCUDAStream()
     │                                      │
     │                                      ├─ 循环从队列取 Kernel
     │                                      │   ├─ kernel->Run()
     │                                      │   └─ kernel.reset()  ← 立即释放，降低峰值显存
     │                                      │
     │                                      └─ 收到 nullptr 哨兵 → 退出
```

### 同步机制：Sync()

```
KernelStream::Sync()
  │
  ├─ 获取互斥锁，等待前一个 Sync 完成
  ├─ 设置 mSharedInfo.sync = true
  ├─ 向 mKernelQueue 推入 nullptr（哨兵 Kernel）
  └─ 等待 mSharedInfo.sync 被清除（AsyncMain 处理哨兵后清除）

AsyncMain 处理哨兵：
  ├─ 调用 SyncImp()
  │   ├─ CpuKernelStream: 空操作（CPU 执行天然有序）
  │   └─ CudaKernelStream: cudaStreamSynchronize()
  └─ 清除 sync 标志 + notify_all
```

### KernelStreamManager — 流管理器

**源文件**: `dtorch/core/kernel_stream/kernel_stream_manager.h/.cc`

`KernelStreamManager` 管理所有流的生命周期：

- 按 `(globalDevice, streamType)` 查找已有流，按 `localDevice` 匹配合适的流实例。
- 如果不存在，通过注册的工厂函数（`CpuKernelStream` 或 `CudaKernelStream`）创建新流。
- 提供 `Sync()` 方法遍历所有流并同步。

## 5. KernelStreamKey — 流的唯一标识

**源文件**: `dtorch/core/type.h`

`KernelStreamKey` 是一个 POD 结构体，由两个字段组成，**唯一标识**一个执行流：

```cpp
struct KernelStreamKey {
    DeviceKey device;            // 设备种类 (kCpu / kGpu) + 设备 ID
    KernelStreamType streamType; // kCompute 或 kCommunicate
};
```

### KernelStreamType 枚举

| 值 | 含义 |
|---|---|
| `kCompute` | 计算流，执行常规算子计算（如 matmul、relu、conv2d） |
| `kCommunicate` | 通信流，执行集合通信操作（如 all-reduce、all-gather） |

> 同一设备（如 GPU 0）可以同时拥有 `kCompute` 和 `kCommunicate` 两个流，从而实现**计算与通信的重叠**。

## 6. OperatorAssignInfo — Operator 到 KernelStream 的映射

**源文件**: `dtorch/core/operators/operator_assign_info.h`

`OperatorAssignInfo` 是 Operator 的一个成员（`Operator::mOperatorAssignInfo`），描述该算子需要在**哪些流**上执行。

### 结构

```cpp
struct OperatorAssignInfo {
    StreamKeySet mStreamKeySet;  // 该算子需要执行的所有流键
    int64_t mMaxGpuId;           // 流键集合中最大的 GPU ID（用于快速查询）

    void Insert(KernelStreamKey& streamKey);
    size_t NumKernelForThisOp() const;  // 返回 mStreamKeySet.size()
    const StreamKeySet& GetStreamKeySet() const;
};
```

- `mStreamKeySet` 中的每个元素对应**一个将要创建的 Kernel**。
- `NumKernelForThisOp()` 即为此算子将创建的 Kernel 总数。

### 填充过程：InferOperatorAssignInfo()

**源文件**: `dtorch/core/operators/operator.cc`（`Operator::InferOperatorAssignInfo()`）

`InferOperatorAssignInfo()` 在 `Operator::Infer()` 流程中被调用，默认逻辑为：

1. 遍历所有**输入 Operand** 的 `DeviceMesh`，为其中的每个设备插入 `KernelStreamKey(device, kCompute)`。
2. 遍历所有**输出 Operand** 的 `DeviceMesh`，为其中的每个设备插入 `KernelStreamKey(device, kCompute)`。

```
示例：一个加法算子，输入/输出 Operand 的 DeviceMesh 包含 {GPU:0, GPU:1}

  OperatorAssignInfo.mStreamKeySet = {
      KernelStreamKey(GPU:0, kCompute),
      KernelStreamKey(GPU:1, kCompute),
  }
  NumKernelForThisOp() = 2   ← 将创建 2 个 Kernel
```

> 部分特殊算子（如 `MemoryOp`）会重写 `InferOperatorAssignInfo()`，从算子参数而非操作数中获取设备列表。

## 7. 完整生命周期

以下以 NaiveRunner 为例，展示从 Operator 到 Kernel 执行的完整流程：

### 阶段 1：推断流分配

```
Operator::Infer()
  └─ InferOperatorAssignInfo()
       └─ 遍历输入/输出 Operand 的 DeviceMesh
            └─ 为每个设备插入 KernelStreamKey(device, kCompute)
                 └─ mOperatorAssignInfo 填充完成
```

### 阶段 2：创建 Kernel

`Runner::CreateKernelForOperator(op)` 在 `naive_runner.cc`：

```
1. 获取 OperatorAssignInfo → StreamKeySet
2. 收集输入 Blob（从 mOperandToBlobs 查找已有 Blob）
3. 创建输出 Blob（为每个输出 Operand 的每个 Device 创建空 Blob）
4. 过滤设备：mSupportedDevices.GetSupported(streamKeySet)
5. 对每个 streamKey：
   ├─ globalDevice = streamKey.GetDevice()
   ├─ localDevice = GlobalToLocal(globalDevice)
   ├─ stream = mStreamManager.GetStream(globalDevice, localDevice, streamType, isAsync=true)
   ├─ 构造 KernelCreateCtx
   ├─ kernel = Kernel::CreateKernel(ctx)
```

### 阶段 3：分派 Kernel

`Runner::Execute()` 在 `naive_runner.cc:40-60`：

```
对每个 Operator：
  ├─ CreateKernelForOperator(op) → 返回 vector<unique_ptr<Kernel>>
  └─ 对每个 Kernel：
       └─ kernel->GetStream().LaunchKernel(std::move(kernel))
            └─ 推入 KernelStream 的无锁队列，立即返回（异步）
```

### 阶段 4：异步执行

在 `KernelStream::AsyncMain()` 线程中：

```
循环：
  ├─ 从 mKernelQueue 取出 Kernel
  ├─ kernel->Run()
  │    ├─ 从 Blob 提取 torch::Tensor 输入
  │    ├─ Compute() → Operator::Compute() → LibTorch 算子（或 [Python Kernel 路径](python_kernel.md)）
  │    └─ 将 torch::Tensor 输出写回 Blob
  └─ kernel.reset()  ← 释放 Kernel 和 Blob 引用
```

### 阶段 5：取值（GetTensorOp）

取值操作被建模为系统算子 `GetTensorOp`：用户请求 Tensor 值时，框架创建 `GetTensorOp`（携带 `TensorPromise`）加入计算图，与普通算子一样被转为 Kernel 在 KernelStream 上执行：

```
GetTensorOp::Compute()
  ├─ 从输入 Blob 提取 torch::Tensor
  ├─ (GPU) 在当前 stream 记录 DeviceEvent → 投递到 BoostAsioThreadPool 后台线程
  │        └─ event->Synchronize()（只等该 stream，而非整卡）→ 在普通 CPU 线程上
  │           完成 CUDA IPC（跨进程传递显存句柄只能在非 CUDA 线程进行）
  └─ promise->SetValue(tensor)  ← 唤醒阻塞在 TensorFuture::Get() 的调用方
```

纯同步（`Graph::Sync()`）同理，建模为零输入零输出的系统算子 `SyncOp`，每个目标设备对应一个 `SyncKernel`，通过 Event + 后台线程异步等待 GPU 完成后设置 `VoidPromise`。详见 [异步获取 Tensor 值](async_get_tensor.md)。


## 8. 关键设计要点

### 8.1 一个 Operator → 多个 Kernel

一个 Operator 对应 `NumKernelForThisOp()` 个 Kernel。对于分布式场景，如果 DeviceMesh 包含 N 个设备，则默认生成 N 个 Kernel，每个 Kernel 绑定一个设备的 `kCompute` 流。

### 8.2 计算与通信分离

`KernelStreamType::kCompute` 和 `kCommunicate` 的分离允许同一设备上同时运行计算和通信 Kernel，通过 CUDA Stream 的并发性实现**计算-通信重叠**（overlap）。

### 8.3 异步三级流水线

```
Client (Python)          Controller (C++)           Worker (KernelStream 线程)
     │                        │                           │
     ├─ 创建 Operator ──────→ │                           │
     │   (异步，立即返回)       ├─ InferOperatorAssignInfo  │
     │                        ├─ CreateKernelForOperator  │
     │                        ├─ LaunchKernel ──────────→ │
     │                        │   (异步，立即返回)          ├─ AsyncMain 循环
     │                        │                           ├─ Kernel::Run()
     │                        │                           └─ 写回 Blob
     │                        │                           │
     ├─ 获取 Tensor 值 ────→  ├─ Sync() ───────────────→ ├─ 等待所有 Kernel 完成
     │   (同步等待)            │   (同步等待)               │
```

三级流水线中，仅当 Client 需要获取 Tensor 值时才会触发同步等待，其余阶段全部异步执行。

### 8.4 Blob 的生命周期管理

- Blob 由 `Runner` 持有（通过 `mOperandToBlobs` 映射），在 Operand 的生命周期内持续存在。
- Kernel 仅持有 Blob 的引用（`std::optional<Blob>`），不拥有所有权。
- Kernel 执行完毕后立即 `reset()`，释放对 Blob 的引用，使不再需要的张量内存可以被回收。

## 9. 源文件索引

| 组件 | 头文件 | 实现文件 |
|---|---|---|
| KernelStreamKey / KernelStreamType | `dtorch/core/type.h` | — |
| OperatorAssignInfo | `dtorch/core/operators/operator_assign_info.h` | — |
| Blob | `dtorch/core/blob.h` | `dtorch/core/blob.cc` |
| Kernel / KernelCreateCtx | `dtorch/core/kernel/kernel.h` | `dtorch/core/kernel/kernel.cc` |
| KernelStream | `dtorch/core/kernel_stream/kernel_stream.h` | `dtorch/core/kernel_stream/kernel_stream.cc` |
| CpuKernelStream | `dtorch/core/kernel_stream/cpu_kernel_stream.h` | — |
| CudaKernelStream | `dtorch/core/kernel_stream/cuda_kernel_stream.h` | `dtorch/core/kernel_stream/cuda_kernel_stream.cc` |
| KernelStreamManager | `dtorch/core/kernel_stream/kernel_stream_manager.h` | `dtorch/core/kernel_stream/kernel_stream_manager.cc` |
| Runner (NaiveRunner) | `dtorch/core/runner/node_runner_base.h` | `dtorch/core/runner/naive_runner.cc` |

## 10. 相关文档

- [设计理念](../design_concept.md) — Eager Graph Architecture 三大核心设计
- [LogicalGraph 计算图表示](logical_graph_representation.md) — Operand / Operator / LogicalGraph 的 DAG 结构
- [Operator 算子体系](../operator/operator.md) — Operator 基类、模板方法模式与完整生命周期
- [Python Kernel](python_kernel.md) — 在 C++ Kernel 中调用 Python 代码（GIL 管理、CUDA Stream 保护、类型转换）
