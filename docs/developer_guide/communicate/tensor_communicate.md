# 集合通信组件

DTorch 的分布式执行需要跨设备、跨线程传输张量数据。`dtorch/core/communication/` 目录提供了三套通信机制——**ThreadGroup**（集合通信原语）、**TensorStore**（跨进程张量存储与交换）和 **TensorPromise/TensorFuture**（异步取值机制）——它们是 Kernel 实现 DeviceMesh 变换（Scatter、Gather、Redistribute）和跨设备数据拷贝的基础。

## 1. 架构概览

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            Kernel (ConvertKernel / CopyKernel / …)                   │
│                                                                                      │
│   ┌──────────────────────────┐  ┌─────────────────────────────────┐  ┌────────────┐  │
│   │  ThreadGroup             │  │  TensorStore                    │  │ Tensor     │  │
│   │  (集合通信原语)            │  │  (进程间张量存储)                 │  │ Promise /  │  │
│   │                          │  │                                 │  │ Future     │  │
│   │  AllReduce / AllGather   │  │  SrcSet  / DestGet              │  │ (异步取值)  │  │
│   │  ReduceScatter / AllToAll│  │  Barrier / Reset                │  │            │  │
│   │  ReplicateToShard / …    │  │                                 │  │ SetValue / │  │
│   │                          │  │  ┌─ MemoryTensorStore (同进程)   │  │ Get / Wait │  │
│   │  ┌────────────────────┐  │  │  ├─ FileTensorStore   (跨进程)   │  │            │  │
│   │  │ Backend            │  │  │  └─ NetworkTensorStore(跨机器)   │  │ ┌─ Memory  │  │
│   │  ├─ ProcessGroupNCCL  │  │  │                                 │  │ ├─ File    │  │
│   │  └─ SameDeviceBackend │  |  |                                 │  │ └─ Network │  │
│   │  │                    │  │  │                                 │  │            │  │
│   │  └────────────────────┘  │  └─────────────────────────────────┘  └────────────┘  │
│   └──────────────────────────┘                                                       │
│                                                                                      │
│   ThreadGroupManager ──► 查找/创建 ThreadGroup ──► 按 ThreadGroupKey 缓存              │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**核心思想**：

- **ThreadGroup** 提供上层集合通信 API（AllReduce、AllGather 等），是 Kernel 进行分布式通信的入口。
- **c10d::ProcessGroupNCCL** 是 ThreadGroup 的另一种后端实现，继承自 `c10d::Backend`。它封装了 NCCL 库，用于不同 GPU 间的跨设备集合通信。
- **ThreadGroupSameDeviceBackend** 是 ThreadGroup 的一种后端实现，用于同一块 GPU 在不同线程/进程间的通讯，其将集合通信原语全部委托给 TensorStore。
- **TensorStore** 是底层跨进程张量交换机制，提供生产者-消费者模型：一个线程 `SrcSet`（写入），其他线程 `DestGet`（读取）。
- **TensorPromise / TensorFuture** 是异步取值机制，基于 Promise-Future 模式：`GetTensorOp` 在 Worker 端通过 `SetValue` 写入张量值，Client 端通过 `Get`/`Wait` 阻塞获取结果，支持同进程（`std::promise`）和跨进程（Boost IPC 共享内存）两种场景。

---

## 2. ThreadGroup — 集合通信体系

ThreadGroup 为 DTorch 提供完整的集合通信能力——同时支持同一进程内多线程、同一机器多进程、以及跨机器的集合通讯。其由四个核心组件构成：**ThreadGroup**（高层 API 封装）、**c10d::ProcessGroupNCCL**（NCCL 后端）、**ThreadGroupSameDeviceBackend**（同设备后端）、**ThreadGroupManager**（生命周期管理）。

### 2.1 ThreadGroup — 集合通信原语

**源文件**:

- `dtorch/core/communication/thread_group/thread_group.h`
- `dtorch/core/communication/thread_group/thread_group.cc`

`ThreadGroup` 是集合通信的**高层 API 封装**。它将一组设备上的线程组成一个通信组，提供常见的集合通信原语(同时支持**同一个进程上的不同线程**和**不同进程上的不同线程**)。所有方法都封装为同步调用（内部调用 `synchronize()` 等待通信完成）。

`ThreadGroup` 内部持有一个 `std::unique_ptr<c10d::Backend>` 指针，根据设备拓扑自动选择后端实现：

- **不同 GPU 间** → `c10d::ProcessGroupNCCL`（PyTorch 标准 NCCL 后端）
- **同一 GPU 上** → `ThreadGroupSameDeviceBackend`（基于 TensorStore 的自定义后端）

两者都继承自 PyTorch 的 `c10d::Backend` 基类，因此 `ThreadGroup` 的集合通信原语（AllReduce、AllGather 等）无需关心底层后端差异——统一通过 `c10d::Backend` 接口调用，多态分发到具体实现。

**接口**：

```cpp
class ThreadGroup {
public:
    ThreadGroup(const std::string& initPath, DeviceKind deviceKind,
                const std::vector<int64_t>& allGlobalDeviceId,
                int rank, int size, bool sameDevice);

    int GetRank() const;          // 本线程在通信组中的 rank
    int GetWorldSize() const;     // 通信组大小
    int64_t GetGlobalDeviceId() const;  // 本 rank 对应的全局设备 ID
    void SetStream(DeviceStream& stream);  // 设置 CUDA Stream
    void Barrier();               // 全局同步

    // P → R：所有 rank 聚合到完全体（all-reduce）
    torch::Tensor AllReduce(torch::Tensor& input, ReduceOpType reduceOpType = kSum);

    // SO → R：将 Shard(0) 的切分张量 all-gather 为单个张量
    torch::Tensor AllGatherIntoTensor(torch::Tensor& input, const Operand& inputOperand);
    torch::Tensor EqualShapeAllGatherIntoTensor(torch::Tensor& input);

    // S → R：all-gather，支持不均匀切分
    torch::Tensor AllGather(torch::Tensor& input, const Operand& inputOperand, int64_t shardIndex);

    // S → R：all-gather 为向量形式，保留各 rank 的独立张量
    std::vector<torch::Tensor> AllGatherIntoVec(torch::Tensor& input, const Operand& inputOperand);

    // P → S0：reduce-scatter
    torch::Tensor ReduceScatterTensor(torch::Tensor& input, ReduceOpType reduceOpType = kSum);

    // S → S：all-to-all 交换
    torch::Tensor AllToAll(torch::Tensor& input, size_t srcDim, size_t destDim,
                           const Operand& inputOperand);

    // R → S：将完整的复制张量均匀切分，取本 rank 对应的分片
    torch::Tensor ReplicateToShard(torch::Tensor& input, int64_t shardIdx,
                                   int64_t subSplitCoordinates = -1);
};
```

### 2.2 c10d::ProcessGroupNCCL — NCCL 后端

`c10d::ProcessGroupNCCL` 是 PyTorch 提供的标准 NCCL 集合通信后端，继承自 `c10d::Backend`。它在 `ThreadGroup` 构造时被选用当 `sameDevice=false`（即通信组内的设备位于不同 GPU 上），利用 NCCL 库实现跨 GPU 的高带宽集合通信。

**后端选择逻辑**（`ThreadGroup::ThreadGroup()`）：

```cpp
if (sameDevice) {
    mImplPtr->backend.reset(new ThreadGroupSameDeviceBackend(initPath, deviceKind, rank, size));
} else {
    auto store = c10::make_intrusive<c10d::FileStore>(initPathInTmpDir, size);
    switch (deviceKind) {
        case DeviceKind::kGpu:
            mImplPtr->backend.reset(new c10d::ProcessGroupNCCL(store, rank, size, opts));
            break;
        default:
            DUnimplemented();
    }
}
```

- 使用 `c10d::FileStore` 作为单机多进程的 rendezvous 机制；多机场景使用 `c10d::TCPStore`(目前暂未支持多机场景)。
- 超时时间由 `GlobalOption::GetSingleton().GetCommTimeoutSecond()` 配置。
- 析构时通过 `ProcessGroupNCCL::shutdown()` 优雅关闭 NCCL Communicator。

`c10d::ProcessGroupNCCL` 提供了标准的 `c10d::Backend` 接口实现：`allreduce`、`allgather`、`_allgather_base`、`_reduce_scatter_base`、`alltoall`、`barrier`、`broadcast`、`send`、`recv` 等，底层通过 NCCL 的 `ncclAllReduce`、`ncclAllGather`、`ncclReduceScatter` 等 API 在 GPU 间高效传输数据。

### 2.3 ThreadGroupSameDeviceBackend — 基于 TensorStore 的后端

**设计动机**：当多个 DTensor Shard 位于同一 GPU 时，NCCL 不支持这种同 GPU 上的集合通信。`ThreadGroupSameDeviceBackend` 后端应用于"单GPU调试分布式程序"的场景：TP、CP 创建的多个进程都绑定到同一块 GPU 上。

`ThreadGroupSameDeviceBackend` 继承自 PyTorch 的 `c10d::Backend`，为**同一物理设备上的多线程**提供集合通信能力。它将所有集合通信操作（broadcast、allreduce、allgather 等）翻译为对 `TensorStore` 的一系列 `SrcSet` / `DestGet` 调用。

**核心实现：以 AllReduce 为例**（`thread_group_same_device_backend.cc:111-146`）：

```cpp
c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
    Sync();
    for (size_t i = 0; i < tensors.size(); i++) {
        at::Tensor result;
        std::string resultStr = "result";
        std::string rankStr = std::to_string(mRank);
        DDebugAssert(mStream != nullptr);

        // 阶段 1：所有 rank 将本地张量写入 TensorStore（每个 rank 1 个消费者）
        mTensorStore->SrcSet(rankStr, tensors[i], *mStream, 1);
        // 阶段 2：rank 0 收集所有数据并求和
        if (mRank == 0) {
            DDebugAssert(opts.reduceOp == c10d::ReduceOp::SUM);
            std::string rankIdxStr = "0";
            result = mTensorStore->DestGet(rankIdxStr, *mStream);
            mTensorStore->DestFinishGet(rankIdxStr, *mStream);
            for (int rankIdx = 1; rankIdx < mWorldSize; rankIdx++) {
                rankIdxStr = std::to_string(rankIdx);
                result += mTensorStore->DestGet(rankIdxStr, *mStream);
                mTensorStore->DestFinishGet(rankIdxStr, *mStream);
            }
        }
        // 等待 rank 0 读取完
        mTensorStore->SrcWaitUntilGetFinished(rankStr, *mStream);

        // 阶段 3：rank 0 广播结果给所有 rank
        if (mRank == 0) {
            tensors[i] = result;
            mTensorStore->SrcSet(resultStr, result, *mStream,
                                static_cast<size_t>(mWorldSize - 1));
            mTensorStore->SrcWaitUntilGetFinished(resultStr, *mStream);
        } else {
            tensors[i] = mTensorStore->DestGet(resultStr, *mStream);
            mTensorStore->DestFinishGet(resultStr, *mStream);
        }

        Reset();
    }
    return c10::make_intrusive<WorkSameDevice>();
}
```

AllReduce 的三阶段展示了 TensorStore 完整的使用模式：

1. **SrcSet** — 写入本地数据，`getCount` 控制需要多少个消费者。
2. **DestGet** + **DestFinishGet** — 消费者读取数据后必须调用 `DestFinishGet` 通知生产者，递增已读取计数。
3. **SrcWaitUntilGetFinished** — 生产者阻塞等待所有消费者完成读取。
4. **Reset** — 清除本次通信的临时数据（内部包含前后两次 `Barrier` + `TensorStore::Reset()`）。

**源文件**:

- `dtorch/core/communication/thread_group/thread_group_same_device_backend.h`
- `dtorch/core/communication/thread_group/thread_group_same_device_backend.cc`

### 2.4 ThreadGroupManager — 生命周期管理

`ThreadGroupManager` 由 `NaiveRunner` 在初始化时创建，负责按需创建和缓存 `ThreadGroup`。不同 Kernel 会共享同一个 `ThreadGroupManager`。

**结构**：

```cpp
class ThreadGroupManager {
    // 按 ThreadGroupKey 缓存所有 ThreadGroup 实例
    std::unordered_map<ThreadGroupKey,
                       std::vector<std::unique_ptr<ThreadGroup>>> mThreadGroupMap;
    // 每个 ThreadGroupKey 对应的初始化字符串（用于 TensorStore 的 storeKey）
    std::unordered_map<ThreadGroupKey, std::string> mThreadGroupInitStringMap;
};
```

**ThreadGroupKey — 通信组标识**：

```cpp
struct ThreadGroupKey {
    DeviceKind deviceKind;        // 设备类型 (kCpu / kGpu)
    int64_t allDeviceId[12];     // 通信组包含的所有设备 ID（最多 12）
};
```

每个不同的设备组合（例如 GPU: [0,1,2,3] 或 GPU: [4,5,6,7]）对应一个唯一的 `ThreadGroupKey`，进而对应一个唯一的 `ThreadGroup`。

**查找/创建逻辑**：

```cpp
ThreadGroup& ThreadGroupManager::GetThreadGroup(DeviceKind deviceKind,
                                                 const std::vector<int64_t>& allDeviceId,
                                                 int64_t currentDeviceId) {
    // 1. 计算 rank（currentDeviceId 在 allDeviceId 中的位置）
    // 2. 构建 ThreadGroupKey
    // 3. 如果 key 不存在，创建 initString 并插入空槽位
    // 4. 如果本 rank 的 ThreadGroup 未创建，则新建并 Barrier
    // 5. 返回本 rank 的 ThreadGroup 引用
}
```

**销毁顺序保证**：析构函数按 `ThreadGroupKey` 的字符串排序后依次销毁所有 `ThreadGroup`。这是为了防止 NCCL 的 `ncclCommDestroy` 在不同 rank 上以不同顺序调用而导致死锁。

**源文件**:

- `dtorch/core/communication/thread_group/thread_group_manager.h`
- `dtorch/core/communication/thread_group/thread_group_manager.cc`

### 2.5 Kernel 中的使用

`ConvertKernel::ConvertPlacements` 是 `ThreadGroup` 的主要使用者，负责 DTensor 的 Placement 重分布。它通过 `ThreadGroupManager` 获取通信子组，根据源和目标的 Placement 类型执行对应的集合通信操作：

```cpp
void ConvertKernel::ConvertPlacements(const TorchTensorOptArray& inputs,
                                       TorchTensorArray& outputs) {
    // 1. 确定差异维度
    auto diffDims = srcPlacementSeq.GetDiffDims(destPlacementSeq);  // 只有一个维度不同

    // 2. 获取通信子组
    ThreadGroupInfo info(srcDeviceMesh.GetMesh(), diffDim, mGlobalDevice.deviceId);
    ThreadGroup& threadGroup = mThreadGroupManager->GetThreadGroup(
        deviceKind, info.GetAllDeviceIds(), mGlobalDevice.deviceId);

    // 3. 设置 Stream（使 SameDeviceBackend 知道在哪个 CUDA Stream 上同步）
    DeviceStream deviceStream = GetDeviceStream();
    threadGroup.SetStream(deviceStream);

    // 4. 同步（多线程 NCCL 并发安全）
    threadGroup.Barrier();

    // 5. 根据 Placement 转换类型执行对应集合通信
    if (src.Shard && dest.Shard) {
        output = threadGroup.AllToAll(input, srcDim, destDim, inputOperand);  // S→S
    } else if (src.Shard && dest.Replicate) {
        output = threadGroup.AllGatherIntoTensor(input, inputOperand);        // S→R
    } else if (src.Replicate && dest.Shard) {
        output = threadGroup.ReplicateToShard(input, shardIdx, subSplit);     // R→S
    } else if (src.Partial && dest.Shard) {
        output = threadGroup.ReduceScatterTensor(input);                      // P→S
    } else if (src.Partial && dest.Replicate) {
        output = threadGroup.AllReduce(input);                                // P→R
    }
}
```

支持的 Placement 转换矩阵：

| Src \ Dest    | Shard(i)         | Shard(j) | Replicate | Partial |
| ------------- | ---------------- | -------- | --------- | ------- |
| **Shard(i)**  | —                | AllToAll | AllGather | —       |
| **Replicate** | ReplicateToShard | —        | —         | —       |
| **Partial**   | ReduceScatter    | —        | AllReduce | —       |

---

## 3. TensorStore — 跨线程张量交换

`TensorStore` 是一个抽象基类，定义了一套**生产者-消费者风格的张量交换接口**。多个线程（可能运行在不同设备上）通过 `key` 来标识和交换张量。

> **⚠️ 并发安全**：TensorStore 涉及多线程/多进程间的张量传递，生产者与消费者往往运行在不同的 CUDA Stream 上。必须精心设计同步机制来避免 race condition——包括但不限于：通过 CUDA Event 确保写入在读取之前完成（`StreamWaitEvent`）、通过 `record_stream` 防止 CUDACachingAllocator 提前回收仍在被其他 Stream 使用的显存、以及正确处理 `getCurrentCUDAStream` 在多线程环境下的 thread-local 语义。详细讨论参见  Stream Race Condition](stream_race_condition.md)。

**源文件**:

- `dtorch/core/communication/tensor_store/tensor_store.h` / `.cc`
- `dtorch/core/communication/tensor_store/memory_tensor_store.h` / `.cc`
- `dtorch/core/communication/tensor_store/file_tensor_store.h` / `.cc`
- `dtorch/core/communication/tensor_store/network_tensor_store.h`

### 3.1 接口定义

TensorStore 是生产者-消费者张量交换抽象，允许多个线程/进程通过 key 共享张量并进行同步访问。

**协议概述 — 每次张量交换遵循四步握手：**

```
Producer (1 writer):                       Consumers (N readers, N = getCount):
  1. SrcSet(key, tensor, getCount=N)        3. DestGet(key)         — 阻塞直到张量被设置
  2. SrcWaitUntilGetFinished(key)           4. DestFinishGet(key)   — 通知生产者已完成读取
```

生产者在第 2 步阻塞，直到所有 N 个消费者都调用了 DestFinishGet（第 4 步）。消费者在第 3 步阻塞，直到生产者调用了 SrcSet（第 1 步）。

**线程安全**：所有方法均可从不同线程/进程并发调用。底层实现使用 mutex + condition_variable（kMemory）或 interprocess mutex + condition_variable（kFile）。

**生命周期**：

```cpp
auto store = TensorStore::Create(createInfo);  // 构造函数内部调用 Barrier()
// ... producer/consumer 操作 ...
store->Reset();  // 清空所有 tensor 并重置状态（barrier 同步）
// store 析构
```

**类定义**：

```cpp
class TensorStore {
public:
    // 工厂方法。根据 createInfo.tensorStoreType 创建具体的 TensorStore 实例。
    // 各实现的构造函数内部会调用 Barrier()，因此所有参与者必须并发（或短时间窗口内）
    // 调用 Create() 以避免死锁。
    static std::shared_ptr<TensorStore> Create(const TensorStoreCreateInfo& createInfo);

    // --- 生产者端 API ---

    // 以 key 发布一个张量，供消费者通过 DestGet() 获取。
    //   key             — 张量在当前 store 会话中的唯一名称。
    //   value           — 要共享的张量，必须位于 stream 对应的设备上。
    //   stream          — value 被生产时所在的 CUDA/CPU stream。内部会在此 stream 上
    //                     记录 event，以便消费者正确同步。
    //   getCount        — 生产者解除阻塞前预期的 DestGet() 调用次数。
    //   destGetDeviceKind — 可选的跨设备传输优化提示。当设置且与源设备类型不同时，
    //                     张量在存储前会被移至目标设备类型（如 CPU→GPU，以便 FileTensorStore
    //                     使用 CUDA IPC）。对于 GPU 张量，CUDA IPC / NCCL send/recv 比
    //                     CPU 序列化/反序列化性能更优。
    virtual void SrcSet(const std::string& key, const torch::Tensor& value,
                        DeviceStream& stream, size_t getCount,
                        std::optional<DeviceKind> destGetDeviceKind = std::nullopt) = 0;

    // 阻塞调用线程，直到所有 getCount 个消费者都对该 key 调用了 DestFinishGet()。
    // 返回后，生产者可以安全地复用或释放张量内存。
    // stream 参数当前实现中未使用，保留用于未来扩展（如基于 stream 的同步）。
    virtual void SrcWaitUntilGetFinished(const std::string& key, DeviceStream& stream) = 0;

    // --- 消费者端 API ---

    // 获取与 key 关联的张量。阻塞直到生产者对该 key 调用了 SrcSet()。
    // 返回的张量通过 SrcSet() 期间记录的内部 event 与生产者 stream 同步，
    // 因此可以安全地在调用 stream 上立即使用。
    // 多个消费者可以并发地对同一个 key 调用 DestGet()。
    virtual torch::Tensor DestGet(const std::string& key, DeviceStream& stream) = 0;

    // 通知生产者本消费者已完成张量使用。每次 DestGet() 调用必须恰好对应一次调用。
    // 当所有 getCount 个消费者都调用了 DestFinishGet() 后，生产者从
    // SrcWaitUntilGetFinished() 中解除阻塞。
    // 对于 GPU 张量，此方法还会对源张量调用 c10::cuda::CUDACachingAllocator::recordStream()，
    // 防止在消费者 stream 完成之前过早回收内存。
    virtual void DestFinishGet(const std::string& key, DeviceStream& stream) = 0;

    // 清空所有已存储张量并重置内部状态。这是一个 barrier 同步操作：
    // 在清空前和后各调用一次 Barrier()，因此所有参与者必须并发调用 Reset()。
    virtual void Reset() = 0;

    // 同步屏障。阻塞直到所有 worldSize 个参与者都调用了 Barrier()。
    // 内部由构造函数和 Reset() 使用，确保所有参与者从一致的状态开始。
    virtual void Barrier() = 0;

    // 便捷辅助方法：调用 DestGet() 后将返回的张量移动到 targetDevice。
    // 处理跨 stream 同步——在 DestGet stream 上记录 event，
    // 在调用 tensor.to(device) 之前在源张量的当前 stream 上等待。
    torch::Tensor DestGetAndToDevice(const std::string& key, DeviceStream& stream,
                                     const Device& targetDevice);
};
```

### 3.2 生产者-消费者模型

`TensorStore` 的核心通信模式如下：

```
                                key = "my_tensor"

    Producer Thread                          Consumer Thread 1
    (SrcSet)                                 (DestGet)
         │                                        │
         ├─ SrcSet("my_tensor", tensor,           |
         │          stream, getCount=2)           │
         │    └─ 写入 tensor，记录 event            │
         │    └─ getCount=2 表示 2 个消费者         │
         │                                        |
         │                                        │
         │                                        ├─ DestGet("my_tensor", stream)
         │                                        │     └─ 等待 Producer 写入完成
         │                                        │     └─ 返回 tensor
         │                                        │
         │                                        │
         │                                        ├─ DestFinishGet("my_tensor", stream)
         │                                        │     └─ 通知 Producer 已读取
         │                                        │
         │                                        │
         ├─ SrcWaitUntilGetFinished(...)          │
         │    └─ 等待 actualGetCount == 2          │
         │    └─ 所有 get 事件完成后返回             │

    Reset() — 清空所有 map，前后各执行一次 Barrier
```

关键同步语义：

- **SrcSet**：生产者将张量存入 key 对应的槽位，记录 CUDA Event，然后通知等待的消费者。
- **DestGet**：消费者阻塞等待直到 key 对应的张量可用；读取时通过 CUDA Event 确保生产者的写入已完成。
- **DestFinishGet**：消费者记录自己的 CUDA Event，递增已读取计数；当 `actualGetCount == targetGetCount` 时通知生产者。
- **SrcWaitUntilGetFinished**：生产者阻塞等待直到所有消费者都已完成读取；然后将所有消费者的 CUDA Event 同步到生产者的 CUDA Stream。

### 3.3 三种存储后端

根据线程间的关系，`TensorStore` 在构造时自动选择后端：

| 后端                   | TensorStoreType | 适用场景         | 底层机制                                                           |
| ---------------------- | --------------- | ---------------- | ------------------------------------------------------------------ |
| **MemoryTensorStore**  | `kMemory`       | 同一进程内多线程 | `std::unordered_map` + `std::condition_variable` + CUDA Event 同步 |
| **FileTensorStore**    | `kFile`         | 同一机器多进程   | 文件系统 + Boost 进程间通信 + CUDA IPC                             |
| **NetworkTensorStore** | `kNetwork`      | 多机器间（预留） | 网络通信 + NCCL send/recv                                          |

**后端选择逻辑**（`ThreadGroupSameDeviceBackend::InitTensorStore()`）：

通过 `IsSameProcessChecker` 判断所有通信方是否在同一进程内：将每个 rank 的 PID 写入共享内存，检查是否存在不同 PID。同进程则使用 `MemoryTensorStore`，否则使用 `FileTensorStore`。（暂不支持 NetworkTensorStore）

### 3.4 MemoryTensorStore 实现细节

**源文件**: `dtorch/core/communication/tensor_store/memory_tensor_store.h` / `.cc`

`MemoryTensorStore` 是同一进程内多线程场景的后端，基于 `std::mutex` + `std::condition_variable` + CUDA Event 实现。

#### 3.4.1 数据结构与实例共享

所有属于同一 `storeKey` 的 `MemoryTensorStore` 实例共享一个 `Impl` 对象，这是多线程通信的核心：

```cpp
struct MemoryTensorStore::Impl {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<std::string, torch::Tensor> tensorMap;            // key → 张量
    std::unordered_map<std::string, std::shared_ptr<DeviceEvent>> setEventMap;  // 写入完成事件
    std::unordered_map<std::string, DeviceStream> setEventStreamMap;     // 写入 stream
    std::unordered_map<std::string, size_t> targetGetCountMap;           // 预期消费者数
    std::unordered_map<std::string, size_t> actualGetCountMap;           // 实际已完成消费者数
    int worldSize;        // 总参与线程数
    int barrierCounts;    // 当前轮次已到达的线程数
    int barrierRound;     // 轮次序号
};
```

`MemoryTensorStoreImpManager` 是全局单例，内部以 `storeKey → weak_ptr<Impl>` 映射管理所有共享实例。`GetImpl(storeKey)` 查找或创建 `Impl`——若已有有效实例则通过 `weak_ptr::lock()` 复用，实现多线程共享；使用 `weak_ptr` 确保所有 `MemoryTensorStore` 析构后 `Impl` 自动回收，每次 `GetImpl` 时调用 `CleanExpiredEntries()` 清理已过期的条目。

构造函数通过 `MemoryTensorStoreImpManager::GetImpl(storeKey)` 获取共享 `Impl`，首次调用时设置 `worldSize`（后续调用校验一致性），随后调用 `Barrier()` 等待所有参与者就绪。

#### 3.4.2 核心方法

- **SrcSet** — 将张量存入 `tensorMap[key]`，记录写入完成时的 CUDA Event 与 stream，设置 `targetGetCount` 并将 `actualGetCount` 清零，释放锁后 `cv.notify_all()` 唤醒消费者。
- **DestGet** — 阻塞等待 `tensorMap` 中存在 key，通过 `setEventMap[key]->StreamWaitEvent(stream)` 确保生产者写入对消费者 stream 可见后返回张量。
- **DestFinishGet** — 递增 `actualGetCount`。对 GPU 张量调用 `CUDACachingAllocator::recordStream()`，告诉 CUDA 缓存分配器该张量在消费者 stream 完成前不可回收显存。当 `actualGetCount == targetGetCount` 时调用 `cv.notify_all()` 唤醒生产者的 `SrcWaitUntilGetFinished`。
- **SrcWaitUntilGetFinished** — 阻塞等待 `actualGetCount == targetGetCount`，返回后生产者可安全复用或释放张量。

#### 3.4.3 Barrier 与 Reset

**Barrier** 采用计数器 + 条件变量实现：`barrierCounts++`，最后到达者重置计数器并递增 `barrierRound` 唤醒所有线程，其余线程阻塞等待轮次变化。

**Reset** 前后各一次 `Barrier` 包裹临界区清空所有 map：第一次 Barrier 确保所有线程完成当前轮次读写，清空后第二次 Barrier 防止线程访问已释放数据。

### 3.5 Kernel 中的使用

`TensorStore` 在多种 Kernel 中被使用，涵盖 DeviceMesh 变换、数据分发/聚合、跨设备拷贝等场景。

#### 3.5.1 Kernel 如何创建 TensorStore

`Kernel` 基类通过 `mTensorStoreCreateInfo` 提供统一的 TensorStore 创建能力：

```cpp
std::shared_ptr<TensorStore> GetTensorStore() const {
    return TensorStore::Create(*mTensorStoreCreateInfo);
}
```

`mTensorStoreCreateInfo` 在 Kernel 创建时（`Runner::CreateKernelForOperator`）由 Runner 设置，其中 `storeKey` 基于 graphId 和 operator 生成唯一标识，确保同一 Operator 的所有 Kernel 共享同一个 TensorStore 命名空间。

当多个 DTorch 实例在同一台机器上运行时，`FileTensorStore` 使用的文件名会冲突。`GlobalCommInstanceId`（`dtorch/core/communication/global_instance_id.h`）为每个 DTorch 实例生成唯一标识符（默认为 PID），确保不同实例的通信命名空间隔离。

#### 3.5.2 ConvertDeviceMesh

`ConvertKernel::ConvertDeviceMesh` 处理张量在不同 DeviceMesh 间的迁移（如从 GPU:0 移到 GPU:1），是 `TensorStore` 生产者-消费者模型的典型应用：

```cpp
void ConvertKernel::ConvertDeviceMesh(const TorchTensorOptArray& inputs,
                                       TorchTensorArray& outputs) {
    auto tensorStore = GetTensorStore();   // 创建新的 TensorStore
    DeviceStream deviceStream = GetDeviceStream();

    // 生产者（输入 Operand 包含此设备）：写入张量
    if (GlobalDeviceInOperand(mOp->OperandX())) {
        std::string key = std::to_string(mGlobalDevice.deviceId);
        DeviceKind destGetDeviceKind = mOp->OperandY()->GetDeviceKind();
        tensorStore->SrcSet(key, inputs[0].value(), deviceStream,
                           /*getCount=*/1, destGetDeviceKind);
    }

    // 消费者（输出 Operand 包含此设备）：读取张量
    if (GlobalDeviceInOperand(mOp->OperandY())) {
        int64_t inputGlobalDeviceId = /* 根据 rank 找源设备 */;
        std::string key = std::to_string(inputGlobalDeviceId);
        output = tensorStore->DestGetAndToDevice(key, deviceStream, mLocalDevice);
        tensorStore->DestFinishGet(key, deviceStream);
    }

    // 生产者等待消费者完成
    if (GlobalDeviceInOperand(mOp->OperandX())) {
        tensorStore->SrcWaitUntilGetFinished(key, deviceStream);
    }
}
```

#### 3.5.3 ScatherTensor 和 GatherTensor

`ScatherTensor` 将本地完整张量按 DeviceMesh 和 Placement 切分后分发到多设备；`GatherTensor` 则反向操作，从多设备收集 Shard 并拼合为完整张量。两者都基于 `TensorStore` 的生产者-消费者模式：

```
ScatherTensor（本地张量 → 分布到多设备）:
  Producer (持有完整张量的线程):
    ├─ DistributedScather: 按 DeviceMesh 和 Placement 切分张量
    └─ 对每个目标设备:
         tensorStore->SrcSet(deviceId, shard_tensor, stream, getCount=1, destDeviceKind)
         tensorStore->SrcWaitUntilGetFinished(deviceId, stream)

  Each Consumer:
    ├─ tensorStore->DestGetAndToDevice(deviceId, stream, localDevice)
    └─ tensorStore->DestFinishGet(deviceId, stream)

GatherTensor（多设备 Shard → 本地拼合）:
  Each Producer:
    ├─ tensorStore->SrcSet(deviceId, local_tensor, stream, getCount=1, destDeviceKind)
    └─ tensorStore->SrcWaitUntilGetFinished(deviceId, stream)

  Consumer (执行 gather 的线程):
    ├─ 从所有 deviceId 读取张量
    ├─ tensorStore->DestFinishGet(deviceId, stream)
    └─ DistributeGather: 按 DeviceMesh 和 Placement 拼合为完整张量
```

#### 3.5.4 CopyKernel

**源文件**: `dtorch/core/kernel/kernel_implement/copy_kernel.h` / `.cc`

`CopyKernel` 实现张量复制操作。当源和目标位于同一 DeviceMesh 时，直接调用 `torch::Tensor::copy_()`；当跨 DeviceMesh 时，使用 TensorStore：

```cpp
void CopyKernel::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& /*outputs*/) {
    if (mNumKernelForThisOp == mOp->OperandA()->GetDeviceMesh().Count()) {
        // 同 DeviceMesh：直接 copy
        inputs[0].value().copy_(inputs[1].value());
    } else {
        // 跨 DeviceMesh：使用 TensorStore
        auto tensorStore = GetTensorStore();
        const std::string key = "tensor";

        // 源设备线程：写入张量
        if (!GlobalDeviceInOperand(mOp->OperandA())) {
            tensorStore->SrcSet(key, inputs[1].value(), deviceStream, /*getCount=*/1, destDeviceKind);
            tensorStore->SrcWaitUntilGetFinished(key, deviceStream);
        }
        // 目标设备线程：读取并复制
        else {
            inputs[0].value().copy_(tensorStore->DestGet(key, deviceStream));
            tensorStore->DestFinishGet(key, deviceStream);
        }
    }
}
```

---

## 4. TensorPromise / TensorFuture — 异步取值机制

`TensorPromise / TensorFuture` 是 DTorch 的**异步取值机制**，基于 Promise-Future 设计模式。它允许 Python Client 在发送异步计算请求后立即获得一个 `TensorFuture` 句柄，随后在任意时刻阻塞等待（或轮询检查）张量值——避免同步停顿破坏异步流水线。

> **扩展**: 除了 `TensorPromise / TensorFuture`，还基于相同的架构实现了 `VoidPromise / VoidFuture`，用于 `Graph::SyncFuture()` 的无数据同步（参见 [async_get_tensor.md](../eager_graph_architecture/async_get_tensor.md)）。同样支持 `kMemory`、`kFile`、`kNetwork` 三种后端，并通过 `VoidFutureCollect` 聚合多个设备的同步 Futures。

**源文件**:

- `dtorch/core/communication/promise_future/tensor_promise_future.h` / `.cc`
- `dtorch/core/communication/promise_future/memory_tensor_promise_future.h` / `.cc`
- `dtorch/core/communication/promise_future/file_tensor_promise_future.h` / `.cc`

### 4.1 设计动机

在 DTorch 的**三级异步流水线**（Client → Controller → Worker）中，当 Client 需要获取某个 Tensor 的结果值时，通过 `_GetTensorAsync` API 发起异步取值操作：

1. **Client 端**：创建一个 `TensorPromise`（根据场景自动选择 Memory 或 File 后端），调用 `GetFuture()` 获取 `TensorFuture`。
2. **Controller**：将携带 Promise 的 `GetTensorOp` 编入计算图，发送到 Worker 执行。
3. **Worker 端**：`GetTensorOp::Compute()` 调用 `promise->SetValue(tensor)` 写入张量值。
4. **Client 端**：调用 `future->Get()` 阻塞等待 Worker 写入完成后返回结果。

### 4.2 类体系与接口

#### 4.2.1 抽象接口

`TensorPromise` 和 `TensorFuture` 分别定义 Promise/Future 的抽象接口：

```cpp
class TensorPromise {
public:
    virtual void SetValue(std::shared_ptr<torch::Tensor> tensor) = 0;  // 写入 Tensor 值
    virtual std::unique_ptr<TensorFuture> GetFuture() = 0;             // 获取关联 Future（仅一次）
    virtual std::string Serialize() const = 0;                         // 序列化（File 返回 shm 文件名）
    virtual void Deserialize(const std::string& data) = 0;             // 反序列化重建
};
```

TensorPromise 需要跨进程传递（单机多进程或未来多机场景），因此必须支持序列化和反序列化：`Serialize()` 将 Promise 的状态编码为字符串（File 模式返回共享内存文件名），`Deserialize()` 从序列化数据重建并 attach 到已有共享内存。Memory 模式无需跨进程，对应方法为空操作。

```cpp
class TensorFuture {
public:
    virtual std::shared_ptr<torch::Tensor> Get() = 0;                         // 阻塞获取
    virtual std::optional<std::shared_ptr<torch::Tensor>> Wait(int64_t) = 0;  // 带超时等待
    virtual bool IsReady() const = 0;                                         // 非阻塞检查值是否就绪
};
```

#### 4.2.2 后端实现

| 后端              | PromiseType | 适用场景         | 底层机制                                                 |
| ----------------- | ----------- | ---------------- | -------------------------------------------------------- |
| **MemoryTensor**  | `kMemory`   | 同一进程内多线程 | `std::promise` / `std::future`，零额外开销，不支持序列化 |
| **FileTensor**    | `kFile`     | 同一机器多进程   | Boost Interprocess 共享内存 + CUDA IPC                   |
| **NetworkTensor** | `kNetwork`  | 多机器间（预留） | —                                                        |

后端选择由 `GetTensorPromiseTypeFromOperand()` 决定：CPU 张量始终使用 Memory 模式；GPU 张量在 `perDevicePerProcess=true` 时使用 File 模式。

#### 4.2.3 工厂函数

三个工厂函数位于 `tensor_promise_future.cc`，封装了 Promise 的创建、重建和类型选择逻辑：

```cpp
// 根据类型创建新 Promise（Client 端在 _GetTensorAsync 中调用）
std::unique_ptr<TensorPromise> CreateTensorPromise(TensorPromiseType type);

// 从序列化数据重建 Promise（Worker 侧反序列化 GetTensorParam 时调用）
// File 模式：创建 FileTensorPromise 并调用 Deserialize(shmFileName) attach 到已有共享内存
// Memory 模式：不应出现（同进程内无需序列化），返回 nullptr
std::unique_ptr<TensorPromise> CreateTensorPromiseFromSerialized(type, data);

// 根据 Operand 确定 Promise 类型
// CPU 张量 → kMemory（CPU 计算始终在进程内通过 NaiveRunner 执行）
// GPU 张量 + perDevicePerProcess → kFile（GPU 子进程需要跨进程传递）
// GPU 张量 + !perDevicePerProcess → kMemory（同进程内通过 std::promise/future 传递）
TensorPromiseType GetTensorPromiseTypeFromOperand(operand, perDevicePerProcess);
```

#### 4.2.4 C++ API 层封装

`dtorch/api/cpp/tensor_future.h` 提供了面向 Python Client 的 `TensorFuture` 封装：

```cpp
class TensorFuture {
public:
    torch::Tensor Get();              // 阻塞获取，内部释放 Python GIL
    torch::Tensor Wait(int64_t ms);   // 带超时等待，超时抛异常
    bool IsReady() const;             // 非阻塞检查值是否就绪
};
```

- 内部持有 `std::unique_ptr<core::communication::TensorFuture>` 和作为生命期守卫的 `Tensor`（防止 Tensor 在 Future 之前析构导致 use-after-free）。
- `Get()` 和 `Wait()` 在执行前通过 `PythonGilScopedRelease` 释放 Python GIL，避免阻塞等待期间阻塞 Python 其他线程。

### 4.3 Memory 后端实现细节

**MemoryTensorPromise** 内部持有 `std::promise<std::shared_ptr<torch::Tensor>>`：

- `SetValue()` → `std::promise::set_value()`
- `GetFuture()` → `std::promise::get_future()` 创建 `MemoryTensorFuture`（仅可调用一次，重复调用触发 `DLogFatal`）
- `Serialize()` / `Deserialize()` 不支持（同进程内无需序列化）

**MemoryTensorFuture** 内部持有 `std::future<std::shared_ptr<torch::Tensor>>`：

- `Get()` → `std::future::get()`（阻塞等待）
- `Wait(timeoutMs)` → `std::future::wait_for()` + 超时检查
- `IsReady()` → `std::future::valid() && std::future::wait_for(0s) == std::future_status::ready` 非阻塞检查就绪状态

> **为什么不用 `std::future::valid()`？** `std::future::valid()` 仅检查 shared state 是否存在（在 DTorch 中始终为 true），不检查值是否就绪。直接用 `valid()` 会导致 `__await__()` 误判值已就绪而跳过轮询，同步调用 `get()` 阻塞 asyncio 事件循环。`IsReady()` 通过 `wait_for(0)` 真正检查值是否就绪，避免此问题。

### 4.4 File 后端实现细节

**FileTensorPromise** 通过 Boost Interprocess 在共享内存中创建通信结构：

```cpp
struct FileTensorPromiseFutureShmImpl {
    // Boost IPC 同步原语
    boost::interprocess::interprocess_mutex mutex;
    boost::interprocess::interprocess_condition cond;
    bool hasValue = false;
    // 通过 TorchUtil::ToIpcMemHandle() 序列化的 tensor 数据
    std::string tensorData;
};
```

- **构造 (Create 模式)**：创建新的共享内存文件，初始化 mutex、cond、hasValue。
- **Deserialize (Attach 模式)**：Worker 侧打开已有共享内存文件，等待 Promise 写入。
- `SetValue()`：将 tensor 通过 `TorchUtil::ToIpcMemHandle()` 序列化后写入共享内存，设置 `hasValue` 标志并通过 `InterprocessCondition` 唤醒等待的 Future。
- `Serialize()`：返回共享内存文件名，用于跨进程序列化时传递给 Worker。

**FileTensorFuture** 打开已有共享内存文件：

- `Get()` / `Wait()`：在 `InterprocessCondition` 上等待，直到 `hasValue` 为 true。
- `IsReady()`：在 mutex 保护下非阻塞检查 `hasValue` 标志是否为 true 且值未被消费。

**跨进程传递**：Promise 的跨进程传递集成在 `GetTensorParam` 的 Boost 序列化中。序列化时（Save）记录 Promise 类型和 `Serialize()` 返回的共享内存文件名；反序列化时（Load）通过 `CreateTensorPromiseFromSerialized()` 重建 Promise 并 attach 到已有共享内存。

### 4.5 使用流程

`_GetTensorAsync` 的完整流程（`tensor_functional.cc:431-460`）：

```cpp
TensorFuture _GetTensorAsync(const Tensor& input) {
    // 1. 若 input 是 DTensor，先 redistribute 为 local tensor
    Tensor localTensor = input;
    if (input.IsDistributed()) {
        DeviceMesh localDevice(Device(input.GetDeviceKind(), 0));
        localTensor = _Redistribute(input, localDevice);
    }

    // 2. 确定 Promise 类型（kMemory / kFile）
    bool perDevicePerProcess = input.GetGraph().GetGraphOption().perDevicePerProcess.value_or(false);
    auto promiseType = GetTensorPromiseTypeFromOperand(*operand, perDevicePerProcess);

    // 3. 创建 Promise → 获取 Future
    auto promise = CreateTensorPromise(promiseType);
    auto future = promise->GetFuture();

    // 4. 创建 GetTensorParam（持有 promise）→ 加入计算图
    auto param = std::make_unique<core::GetTensorParam>(std::move(promise));
    GraphConstructor::AddOperator(std::move(param), {localTensor}, true);

    // 5. 立即返回 TensorFuture（非阻塞）
    return TensorFuture(input, std::move(future));
}
```

关键设计点：

- **Promise 由 GetTensorOp 携带入图**——Promise 的生命期绑定到 Operator 参数中，确保在 Worker 执行 `Compute()` 时仍然存活。
- **Future 立即返回给 Client**——Client 无需等待 Worker 执行完成，可稍后调用 `Get()`/`Wait()` 获取结果。
- **Tensor 作为生命期守卫**——`api::cpp::TensorFuture` 同时持有 Tensor 和内部 Future，防止 Tensor 在 Future 之前析构。

---

## 5. 设计要点

### 5.1 三层通信抽象

DTorch 的通信体系采用三层设计：

- **TensorPromise/TensorFuture**：异步取值机制，解决 "Client 如何非阻塞地获取 Worker 端的张量值"。
- **TensorStore**：底层点对点（key-value 风格）的线程间张量交换，解决 "如何把张量从线程 A 传给线程 B"。
- **ThreadGroup**：上层集合通信语义（AllReduce、AllGather 等），解决 "分布式张量如何在不同 placement 间变换"。

### 5.2 同一设备 vs 不同设备

ThreadGroup 的后端选择是自动的：

- **同设备线程** → `ThreadGroupSameDeviceBackend` → `MemoryTensorStore`（轻量，无 NCCL 开销）
- **不同设备线程** → `ProcessGroupNCCL`（利用 GPU 间的高带宽 NVLink/PCIe）

这保证了 DTorch 即支持单 GPU 多 Stream 场景下的通信（不经过 NCCL），同时保留了多 GPU 场景的性能。

### 5.3 异步友好

TensorStore 的 API 通过 CUDA Event 而非全局同步来实现生产者-消费者同步，允许计算和通信在不同 CUDA Stream 上重叠执行。`SrcSet` 记录事件后即可返回，消费者通过事件等待而非轮询。

---

## 6. 源文件索引

| 组件                                                  | 头文件                                                         | 实现文件                                                        |
| ----------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| TensorPromise / TensorFuture（基类 + 工厂）           | `dtorch/core/communication/promise_future/tensor_promise_future.h`            | `dtorch/core/communication/promise_future/tensor_promise_future.cc`            |
| MemoryTensorPromise / MemoryTensorFuture              | `dtorch/core/communication/promise_future/memory_tensor_promise_future.h`     | `dtorch/core/communication/promise_future/memory_tensor_promise_future.cc`     |
| FileTensorPromise / FileTensorFuture                  | `dtorch/core/communication/promise_future/file_tensor_promise_future.h`       | `dtorch/core/communication/promise_future/file_tensor_promise_future.cc`       |
| TensorStore（基类）                                   | `dtorch/core/communication/tensor_store/tensor_store.h`                     | `dtorch/core/communication/tensor_store/tensor_store.cc`                     |
| MemoryTensorStore                                     | `dtorch/core/communication/tensor_store/memory_tensor_store.h`              | `dtorch/core/communication/tensor_store/memory_tensor_store.cc`              |
| FileTensorStore                                       | `dtorch/core/communication/tensor_store/file_tensor_store.h`                | `dtorch/core/communication/tensor_store/file_tensor_store.cc`                |
| NetworkTensorStore                                    | `dtorch/core/communication/tensor_store/network_tensor_store.h`             | —                                                               |
| ThreadGroup                                           | `dtorch/core/communication/thread_group/thread_group.h`                     | `dtorch/core/communication/thread_group/thread_group.cc`                     |
| ThreadGroupSameDeviceBackend                          | `dtorch/core/communication/thread_group/thread_group_same_device_backend.h` | `dtorch/core/communication/thread_group/thread_group_same_device_backend.cc` |
| ThreadGroupManager / ThreadGroupKey / ThreadGroupInfo | `dtorch/core/communication/thread_group/thread_group_manager.h`             | `dtorch/core/communication/thread_group/thread_group_manager.cc`             |
| GlobalCommInstanceId                                  | `dtorch/core/communication/global_instance_id.h`               | —                                                               |
| GetTensorOp / GetTensorParam                          | `dtorch/core/operators/system/get_tensor_op.h`                 | `dtorch/core/operators/system/get_tensor_op.cc`                 |
| TensorFuture (C++ API 封装)                           | `dtorch/api/cpp/tensor_future.h`                               | `dtorch/api/cpp/tensor_future.cc`                               |
| ConvertKernel                                         | `dtorch/core/kernel/kernel_implement/convert_kernel.h`                          | `dtorch/core/kernel/kernel_implement/convert_kernel.cc`                          |
| CopyKernel                                            | `dtorch/core/kernel/kernel_implement/copy_kernel.h`                             | `dtorch/core/kernel/kernel_implement/copy_kernel.cc`                             |
| DistributedScather / DistributeGather                 | `dtorch/core/kernel/kernel_implement/distributed_scather_and_gather.h`          | `dtorch/core/kernel/kernel_implement/distributed_scather_and_gather.cc`          |

## 7. 相关文档

- [设计理念](../design_concept.md) — Eager Graph Architecture 三大核心设计
- [Kernel 运行时：从 Operator 到 KernelStream 的执行映射](../eager_graph_architecture/kernel_runtime.md) — Kernel 的生命周期、KernelStream 调度
