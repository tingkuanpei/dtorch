# GraphExecutor in Cluster（单机多进程）

[graph_executor.md](graph_executor.md) 介绍了单机多线程场景下的执行管线。本文档介绍 **单机多进程** 场景：当用户设置 `perDevicePerProcess = true` 时，每个 GPU 运行在独立的子进程中，通过 ZMQ IPC 与 Controller 进程通信。

本文档与 [zmq.md](../communicate/zmq.md) 配合阅读——zmq.md 详述通信协议和消息格式，本文档聚焦 Runner 架构和进程管理。

---

## 1. 架构总览

单机多进程场景新增了以下核心组件，形成跨进程的执行管线：

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Parent Process (Controller)                   │
│                                                                      │
│  GraphConstructor → EagerGraphExecutor                               │
│                       │                                              │
│                       ├── RemoteRunnerPublisher (PUB)                │
│                       │   广播 Execute/Destroy                        │
│                       │                                              │
│                       ├── RemoteRunnerPuller (PULL)                  │
│                       │   接收子进程的 devicesReady 通知               │
│                       │                                              │
│                       └── PerDeviceProcessNodeRunner                 │
│                            │                                         │
│                            ├── NaiveRunner (CPU, kFile store)        │
│                            │                                         │
│                            └── RemoteRunnerInProcess[]               │
│                                 │ (每个 GPU 一个)                     │
│                                 │                                    │
│                                 ├── SubProcess subprocess            │
│                                 └── SubProcessSync processSync       │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                    Child Process (per GPU)                            │
│                                                                      │
│  RemoteRunnerInProcessLauncher::BackgroundProcessExecMain()          │
│       │                                                              │
│       └── RemoteRunner                                               │
│            │                                                         │
│            ├── RemoteRunnerPusher (PUSH)                              │
│            │   NotifyDevicesReady: 通知主节点设备已就绪                │
│            │                                                         │
│            ├── RemoteRunnerSubscriber (SUB)                           │
│            │   接收 Publisher 广播的 Execute/Destroy                   │
│            │                                                         │
│            └── NaiveRunner (GPU, kFile store)                        │
│                 KernelStream::LaunchKernel()                         │
└──────────────────────────────────────────────────────────────────────┘
```

对比单机多线程场景（`graph_executor.md` 第 1 节），核心变化：

| 方面 | 单机多线程 (`PerDeviceThreadNodeRunner`) | 单机多进程 (`PerDeviceProcessNodeRunner`) |
|---|---|---|
| **GPU 运行方式** | 主进程内的独立 CUDA Stream 线程 | 独立子进程，每个绑定一张 GPU |
| **通信方式** | 内存共享 (`kMemory` store) | 文件共享内存 (`kFile` store) + ZMQ IPC |
| **Controller-Runner 交互** | 直接函数调用 | PUB-SUB 广播 + PUSH-PULL 就绪通知 |
| **Runner 实现** | `NaiveRunner` 直接执行 | `RemoteRunner`（内含 `NaiveRunner`，SUB + PUSH） |

---

## 2. NodeRunnerBase — Runner 统一接口

`NodeRunnerBase` 是所有 Runner 的抽象基类，定义了 EagerGraphExecutor 与执行层之间的统一契约。

源码位置：`dtorch/core/runner/node_runner_base.h`

```cpp
class NodeRunnerBase {
public:
    virtual void Execute(
        const std::vector<std::shared_ptr<Operator>>& ops,
        const std::vector<const Operand*>& noHoldOperands) = 0;
};
```

单一纯虚函数 `Execute` 负责执行一批 Operator 并释放不再引用的 Operand。

### 2.1 派生类体系

```
NodeRunnerBase (抽象接口)
  ├── PerDeviceThreadNodeRunner      单机多线程，直接委托 NaiveRunner
  └── PerDeviceProcessNodeRunner     单机多进程，管理 CPU Runner + GPU 子进程
```

| 派生类 | 源码位置 | 使用场景 | Execute 实现 |
|---|---|---|---|
| `PerDeviceThreadNodeRunner` | `dtorch/core/runner/per_device_thread_node_runner.h` | `perDevicePerProcess=false`, 单节点 | 委托 `NaiveRunner`（`kMemory` store） |
| `PerDeviceProcessNodeRunner` | `dtorch/core/runner/remote/per_device_process_node_runner.h` | `perDevicePerProcess=true`, 单节点或多节点 | CPU: 委托 `NaiveRunner`；GPU: 通过 `RemoteRunnerPublisher` 广播 |

### 2.2 EagerGraphExecutor 中的 Runner 选择与初始化

`EagerGraphExecutor` 在构造函数中根据 `GraphOption` 和 `ClusterInfo` 决定使用哪种 Runner，并创建通信通道：

```cpp
// 创建 Publisher（PUB 广播）和 Puller（PULL 接收子进程就绪通知）
std::string publisherAddress;
std::string pushPullAddress;
if (graphOption.perDevicePerProcess.value() || mClusterInfo.NodeSize() > 1) {
    publisherAddress = external::zmq::GetRandomZmqIpcAddress();
    pushPullAddress = external::zmq::GetRandomZmqIpcAddress();
    mRemoteRunnerPublisher = std::make_unique<RemoteRunnerPublisher>(publisherAddress);
    mRemoteRunnerPuller = std::make_unique<RemoteRunnerPuller>(pushPullAddress);
}

// InitCurrentNode 根据 perDevicePerProcess 选择 Runner
InitCurrentNode(mGraphOption, ..., supportedDevicesForNodes[0], publisherAddress, pushPullAddress);

// 等待所有远端 device 就绪
WaitAllRunnerReady(supportedDevicesForNodes);
```

关键逻辑：
- 当 `perDevicePerProcess = true` **或** 集群节点数 > 1 时，同时创建 `RemoteRunnerPublisher`（PUB 广播）和 `RemoteRunnerPuller`（PULL 接收就绪通知）。
- `InitCurrentNode` 中：`perDevicePerProcess=true` → `PerDeviceProcessNodeRunner`（子进程）；`perDevicePerProcess=false` → `PerDeviceThreadNodeRunner`（本地线程，无需等待）。
- `WaitAllRunnerReady` 根据 `supportedDevicesForNodes` 计算需要等待的设备集合，通过 `RemoteRunnerPuller::Get()` 轮询，按 device 逐个确认就绪后才继续。

### 2.3 EagerGraphExecutor 的执行分发

`EagerGraphExecutor::ExecuteOperatorsAndNoHoldOperands` 展示了两种通道如何协作：

```cpp
void EagerGraphExecutor::ExecuteOperatorsAndNoHoldOperands(...) {
    // 1. PUB-SUB 广播：发送给所有 GPU 子进程
    if (mRemoteRunnerPublisher) {
        mRemoteRunnerPublisher->Execute(ops, noHoldOperands);
    }

    // 2. 本地执行：CPU Runner 在父进程内运行
    for (auto& runner : mNodeRunners) {
        runner->Execute(ops, noHoldOperands);
    }
}
```

- `mRemoteRunnerPublisher->Execute()` 将算子序列化并广播到所有子进程的 `RemoteRunner`（内 SUB socket 接收）。
- `runner->Execute()` 在父进程内执行 CPU 算子（`PerDeviceProcessNodeRunner` 的 `mCpuRunner`）。

---

## 3. 跨进程通信组件

单机多进程场景引入了以下核心组件，在父进程（Controller）和 GPU 子进程（Worker）之间建立通信通道。

详细的 ZMQ 协议和消息格式见 [zmq.md](../communicate/zmq.md)，Operator 的序列化机制见 [serialization.md](serialization.md)。以下从 Runner 架构的角度介绍各组件的角色。

### 3.1 RemoteRunnerPublisher — 广播发送端

源码位置：`dtorch/external/zmq/remote_runner_publisher.h` / `.cc`

**角色**：PUB-SUB 的发布端，由 `EagerGraphExecutor` 持有。负责将 `Execute`、`Destroy` 两种命令广播给所有 GPU 子进程。

```cpp
class RemoteRunnerPublisher {
    void Execute(ops, noHoldOperands);  // 序列化算子并广播
    void SendDestroy();                 // 通知所有子进程退出
};
```

- 内部绑定一个 PUB Socket，`sndhwm = 0`（无发送队列上限）。
- 每条消息包含单调递增的消息 ID（由 `PublishMessageIdManager` 分配）。

### 3.2 RemoteRunnerPusher — 就绪通知

源码位置：`dtorch/external/zmq/remote_runner_pusher.h` / `.cc`

**角色**：在子进程的 `RemoteRunner` 中使用。子进程初始化完成后调用 `NotifyDevicesReady(devices)`，通过 PUSH socket 通知主节点的 `RemoteRunnerPuller`。

```cpp
class RemoteRunnerPusher {
    static const std::string kDevicesReadyStr;  // "devicesReady"
    void NotifyDevicesReady(const std::vector<Device>& devices);
};
```

使用 Boost 序列化将 device 列表写入消息体，以 2-part ZMQ 消息发送。

### 3.3 RemoteRunnerPuller — 就绪收集

源码位置：`dtorch/external/zmq/remote_runner_puller.h` / `.cc`

**角色**：由 `EagerGraphExecutor` 持有。PULL socket bind 在 `pushPullAddress`，非阻塞 `Get()` 收集子进程的就绪通知。

```cpp
class RemoteRunnerPuller {
    bool Get();                                 // 非阻塞接收一条消息
    const std::string& GetMessageType() const;   // 返回消息类型
    const std::vector<Device>& GetReadyDevice() const; // 返回 ready device 列表
    void Clear();
};
```

`EagerGraphExecutor::WaitAllRunnerReady()` 轮询 `Get()`，收到 `kDevicesReadyStr` 消息后从 `DeviceKeySet` 中逐个 erase，确保每个预期的 device 都已就绪。

### 3.4 RemoteRunner — 子进程执行引擎

源码位置：`dtorch/core/runner/remote/remote_runner.h` / `.cc`

**角色**：在每个 GPU 子进程中运行，整合了 ZMQ 通信和 NaiveRunner 执行：

```cpp
class RemoteRunner {
    // ZMQ 通信:
    RemoteRunnerPusher mPusher;         // PUSH: NotifyDevicesReady 通知主节点
    RemoteRunnerSubscriber mSubscriber; // SUB: 接收 Publisher 广播的 Execute/Destroy

    // 执行引擎:
    NaiveRunner mNaiveRunner;           // 核心执行引擎（kFile store）
    std::unordered_map<uintptr_t, std::shared_ptr<Operand>> mOperandMap;

    // 线程:
    std::thread mAsyncThread;
    std::atomic_bool mGetDestroySignal;
};
```

`AsyncMain` 流程：
1. 等待 SUB 连接建立 → 调用 `mPusher.NotifyDevicesReady()` 通知主节点
2. 循环：`mSubscriber.Get()` 非阻塞轮询 → `kExecuteStr` → `ExecuteSerialization()` → `mNaiveRunner.Execute()` → `kDestroyStr` → 退出

---

## 4. PerDeviceProcessNodeRunner — 单节点进程 Runner

`PerDeviceProcessNodeRunner` 是实现 `perDevicePerProcess` 模式的核心 Runner。它管理一台机器上的所有 CPU 和 GPU 资源。

源码位置：`dtorch/core/runner/per_device_process_node_runner.h` / `.cc`

### 4.1 内部结构

```cpp
class PerDeviceProcessNodeRunner : public NodeRunnerBase {
    GraphOption mGraphOption;
    std::unique_ptr<NaiveRunner> mCpuRunner;          // 所有 CPU 共享
    DeviceKeyMap<RemoteRunnerInProcess> mGpuRemoteRunnerMap;  // 每个 GPU 一个
};
```

- **`mCpuRunner`**：使用 `kFile` 类型的 `TensorStore`（支持跨进程共享内存）。CPU 上的算子直接在父进程内执行。
- **`mGpuRemoteRunnerMap`**：`DeviceKey → RemoteRunnerInProcess` 的映射。每张 GPU 对应一个子进程。

### 4.2 构造过程

```cpp
PerDeviceProcessNodeRunner::PerDeviceProcessNodeRunner(graphOption, supportedDevices,
                                                        publisherAddress, pushPullAddress) {
    // 1. 创建 CPU Runner（kFile store）
    mCpuRunner = std::make_unique<NaiveRunner>(
        graphOption, RunnerSupportedDevices(/*cpu only*/),
        TensorStoreConfig(TensorStoreType::kFile));

    // 2. 为每张 GPU 创建子进程
    for (const auto& devicePair : supportedDevices.AllDevices()) {
        mGpuRemoteRunnerMap.emplace(
            devicePair.globalDevice,
            RemoteRunnerInProcess(graphOption, supportedDevices,
                                  publisherAddress, pushPullAddress));
    }

    // 3. 等待所有子进程启动完成（仅共享内存同步，ZMQ 握手由 Push/Pull 替代）
    for (auto& it : mGpuRemoteRunnerMap) {
        it.second.WaitSubProcessStarted();
    }
}
```

### 4.3 析构清理

```cpp
PerDeviceProcessNodeRunner::~PerDeviceProcessNodeRunner() {
    // Notify all subprocesses to exit concurrently before clearing the map.
    for (auto& it : mGpuRemoteRunnerMap) {
        it.second.NotifySubProcessExit();
    }
    mGpuRemoteRunnerMap.clear();
}
```

子进程在 `BackgroundProcessExecMain` 中通过共享内存条件变量等待退出通知：
- 父进程调用 `NotifySubProcessExit()` → `SubProcessSync::NotifyExit()` 在共享内存中设置 `exitFlag` 并通知条件变量 → 子进程 `WaitForExit()` 返回 → `BackgroundProcessExecMain` 返回 0，进程正常退出
- 子进程在 `BackgroundProcessExecMain` 中调用 `processSync.WaitForExit()` 阻塞等待，父进程通知后继续执行并正常退出

---

## 5. RemoteRunnerInProcess — 子进程管理单元

`RemoteRunnerInProcess` 是一个结构体（非 `NodeRunnerBase` 子类），将子进程管理所需的所有组件打包在一起。

源码位置：`dtorch/core/runner/remote/remote_runner_in_process.h`

```cpp
struct RemoteRunnerInProcess {
    SubProcessSync processSync;
    SubProcess subprocess;          // 子进程句柄，内部持有 RemoteRunner

    RemoteRunnerInProcess(graphOption, supportedDevices,
                          publisherAddress, pushPullAddress);
    void WaitSubProcessStarted();
    void NotifySubProcessExit();    // 通过共享内存通知子进程退出
    ~RemoteRunnerInProcess();       // 通知子进程退出并等待其结束
};
```

### 5.1 成员职责

| 成员 | 类型 | 职责 |
|---|---|---|
| `subprocess` | `SubProcess` | 子进程句柄，管理 OS 进程的生命周期。子进程内部创建了 `RemoteRunner`（内含 `NaiveRunner`） |
| `processSync` | `SubProcessSync` | 通过共享内存在父子进程间传递启动参数、同步启动状态、以及通知子进程退出 |

`publisherAddress` 和 `pushPullAddress` 作为构造函数参数传入，仅在构造时用于传递给子进程，不作为成员变量存储。

### 5.2 构造过程

```cpp
RemoteRunnerInProcess::RemoteRunnerInProcess(graphOption, supportedDevices,
                                              publisherAddress, pushPullAddress)
    : processSync(), subprocess() {
    auto [subProcess, subProcessSync] =
        RemoteRunnerInProcessLauncher::StartRemoteRunnerInBackgroundProcess(
            publisherAddress, pushPullAddress, graphOption, supportedDevices);
    subprocess = std::move(subProcess);
    processSync = std::move(subProcessSync);
}
```

`publisherAddress` 是共享的——所有子进程的 SUB Socket 都连接到同一个 Publisher。`pushPullAddress` 也是共享的——所有子进程的 PUSH socket 连接到同一个 Puller。

### 5.3 WaitSubProcessStarted — 启动同步

```cpp
void RemoteRunnerInProcess::WaitSubProcessStarted() {
    processSync.WaitProcessStarted();  // 等待子进程初始化完成
    // NOTE: processSync is intentionally NOT reset — the shared memory
    // mapping must remain alive so that NotifySubProcessExit() can later
    // signal the child process to exit via the shared memory condition variable.
}
```

仅通过共享内存等待子进程初始化完成。原来的 ZMQ REQ-REP 握手（`client.WaitServerReady()`）已被 PUSH-PULL 就绪通知替代：子进程 `RemoteRunner::AsyncMain` 中调用 `NotifyDevicesReady()` → 主节点 `WaitAllRunnerReady()` 接收。

与旧版本不同，共享内存映射**不再释放**（不调用 `Reset()`），因为析构时需要用它通知子进程退出。

### 5.4 子进程启动

`RemoteRunnerInProcess` 构造时调用 `RemoteRunnerInProcessLauncher::StartRemoteRunnerInBackgroundProcess` 启动子进程，依赖两个基础类：

- **`SubProcess`**（`dtorch/common/process/sub_process.h`）：封装 OS 进程生命周期，通过 `exec` 启动子进程，析构时执行 `process->wait()` 等待子进程退出并检查退出码。退出通知通过共享内存条件变量（`SubProcessSync::NotifyExit()` / `WaitForExit()`）实现。

  > **为什么必须用 `exec` 而非纯 `fork`？** CUDA 不支持在创建 CUDA context 后 `fork` 子进程。因此子进程必须以 `exec` 方式启动一个全新的可执行文件（`dtorch_launcher --remote-runner`），在新进程中重新初始化 CUDA。
- **`SubProcessSync`**（`dtorch/common/process/sub_process_sync.h`）：通过共享内存在父子进程间传递序列化的启动参数（`publisherAddress`、`pushPullAddress`、`graphOption` 等）、同步启动状态（`WaitProcessStarted` / `NotifyProcessStarted`）、以及通知子进程退出（`NotifyExit` / `WaitForExit`）。

启动流程：

1. 父进程序列化参数（Boost BinaryOArchive），写入共享内存（`SubProcessSync`）
2. 父进程 `exec` 子进程（`dtorch_launcher --remote-runner --shm_file_name=<...>`）
3. 子进程读取共享内存 → 反序列化参数 → 创建 `RemoteRunner` → `NotifyProcessStarted()`
4. 父进程 `WaitSubProcessStarted()` → 继续
5. 子进程 `RemoteRunner::AsyncMain` 中 `NotifyDevicesReady()` → PUSH → 主节点 `RemoteRunnerPuller` 接收

---

## 6. 完整执行流程

以下以 `c = a + b` 在单机多进程（2 GPU）场景下的执行流程为例：

```
Python: c = a + b
    │
    ▼
GraphConstructor::AddOperator()
    └── EGEMessageQueue::PushMessage(AddOperatorEEMsg)
    │
    ▼ (异步)
EagerGraphExecutor::AsyncMain()
    ├── GetEagerGraphExecutorMessage() → 消费 AddOperatorEEMsg
    ├── GraphTraversalSequence::FromVec(...)
    │
    ├── ExecuteOperatorsAndNoHoldOperands(traversalSeq, noHoldOperands)
    │   │
    │   ├── mRemoteRunnerPublisher->Execute(ops, noHoldOperands)
        │   │   └── PUB [id=0, "publisherExecute", data]
    │   │       │
    │   │       ├──────────→ GPU0 Child: SUB recv
    │   │       │   RemoteRunner (old path, unchanged)()
    │   │       │   → ProcessSubscriberExecuteMessage()
    │   │       │   → ExecuteSerialization() → Execute()
    │   │       │   → NaiveRunner::Execute()
    │   │       │   → KernelStream::LaunchKernel()  // GPU 0
    │   │       │
    │   │       └──────────→ GPU1 Child: SUB recv
    │   │           (同上)                      // GPU 1
    │   │
    │   └── mNodeRunners[0]->Execute(ops, noHoldOperands)
    │       └── PerDeviceProcessNodeRunner::Execute()
    │           └── mCpuRunner->Execute()  // CPU 算子（如有）
    │
    └── LogicalGraph::DeleteOperator(op)
```


---

## 7. 源码索引

| 组件 | 头文件 | 实现文件 |
|---|---|---|
| `NodeRunnerBase` | `dtorch/core/runner/node_runner_base.h` | — |
| `PerDeviceThreadNodeRunner` | `dtorch/core/runner/per_device_thread_node_runner.h` | — |
| `PerDeviceProcessNodeRunner` | `dtorch/core/runner/remote/per_device_process_node_runner.h` | `dtorch/core/runner/remote/per_device_process_node_runner.cc` |
| `RemoteRunner` | `dtorch/core/runner/remote/remote_runner.h` | `dtorch/core/runner/remote/remote_runner.cc` |
| `RemoteRunnerInProcess` | `dtorch/core/runner/remote/remote_runner_in_process.h` | `dtorch/core/runner/remote/remote_runner_in_process.cc` |
| `SubProcess` | `dtorch/common/process/sub_process.h` | — |
| `SubProcessSync` | `dtorch/common/process/sub_process_sync.h` | — |
| `RemoteRunnerPublisher` | `dtorch/external/zmq/remote_runner_publisher.h` | `dtorch/external/zmq/remote_runner_publisher.cc` |
| `RemoteRunnerSubscriber` | `dtorch/external/zmq/remote_runner_subscriber.h` | `dtorch/external/zmq/remote_runner_subscriber.cc` |
| `RemoteRunnerPusher` | `dtorch/external/zmq/remote_runner_pusher.h` | `dtorch/external/zmq/remote_runner_pusher.cc` |
| `RemoteRunnerPuller` | `dtorch/external/zmq/remote_runner_puller.h` | `dtorch/external/zmq/remote_runner_puller.cc` |
| `PublishMessageIdManager` | `dtorch/external/zmq/remote_runner_publisher.h` | — |
| ZMQ 工具函数 | `dtorch/external/zmq/zmq.h` | — |
| `EagerGraphExecutor` | `dtorch/core/graph/eager_graph_executor.h` | `dtorch/core/graph/eager_graph_executor.cc` |
| `GraphOption` | `dtorch/api/cpp/graph.h` | — |

相关文档：
| 文档 | 内容 |
|---|---|
| [zmq.md](../communicate/zmq.md) | ZMQ 通信协议与消息格式详解 |
| [graph_executor.md](graph_executor.md) | 单机多线程场景的执行管线 |
| [tensor_communicate.md](../communicate/tensor_communicate.md) | ThreadGroup / TensorStore 集合通信机制 |
| [kernel_runtime.md](kernel_runtime.md) | Kernel / KernelStream / Blob 运行时执行 |
