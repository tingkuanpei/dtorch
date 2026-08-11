# Eager Graph Engine

DTorch 的 Eager Graph 执行引擎由四层核心组件构成，负责将用户 API 调用转换为可执行的 Kernel 并在设备上调度执行。本文档介绍单机多线程场景下的执行管线：**GraphConstructor** → **EagerGraphExecutor** → **PerDeviceThreadNodeRunner** → **NaiveRunner**。

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Python API 层                                 │
│  api::cpp::functional::add(...) / tensor_a + tensor_b                │
├──────────────────────────────────────────────────────────────────────┤
│                    GraphConstructor                                  │
│  Operator 工厂 → 引用计数管理 → 消息序列化 → EGEMessageQueue           │
├──────────────────────────────────────────────────────────────────────┤
│                    EagerGraphExecutor                                │
│  AsyncMain 消息循环 → LogicalGraph 构建 → GraphTraversalSequence      │
│  → RunnerBase::Execute()                                             │
├──────────────────────────────────────────────────────────────────────┤
│                    PerDeviceThreadNodeRunner (Runner)                │
│  委托 NaiveRunner 执行                                                │
├──────────────────────────────────────────────────────────────────────┤
│                    NaiveRunner (执行引擎)                             │
│  Operator → Kernel 转换 → Blob 管理 → KernelStream::LaunchKernel()    │
└──────────────────────────────────────────────────────────────────────┘
```

核心数据流：

1. **GraphConstructor** 接收 Python API 调用，通过 `OperatorFactory` 创建 `Operator`，封装为 `EGEMessage` 推入消息队列。
2. **EagerGraphExecutor** 的 `AsyncMain` 线程从消息队列消费消息，将 Operator 加入 `LogicalGraph`，按拓扑序遍历并调用 Runner 执行。
3. **PerDeviceThreadNodeRunner** 作为 `RunnerBase` 的线程级实现，将所有调用委托给内部的 `NaiveRunner`。
4. **NaiveRunner** 将 Operator 转换为 Kernel，管理 Blob 物理内存，通过 `KernelStream` 将 Kernel 发射到 CUDA Stream 或 CPU 线程执行。

---

## 2. GraphConstructor — 计算图构造器

`GraphConstructor` 是 Python API 层与 C++ 执行引擎之间的**桥梁**。用户通过 `api::cpp::functional` 或 Tensor 运算符发起调用时，GraphConstructor 负责将调用转换为 Operator 并异步发送给 EagerGraphExecutor。

源码位置：`dtorch/core/graph/graph_constructor.h` / `.cc`

### 2.1 核心职责

| 职责 | 说明 |
|---|---|
| **Operator 构造** | 通过 `OperatorFactory` 将 `OpParam` + 输入 Operand 实例化为 Operator |
| **引用计数管理** | 通过 `OperandCache::apiTensorRefCount` 追踪 Python 侧 Tensor 对 Operand 的引用 |
| **消息序列化** | 将 Operator、Operand 生命周期事件封装为 `EGEMessage`，推入 `EGEMessageQueue` |
| **同步等待** | 提供 `Sync()` 和 `GetSharedPtrTensor()` 接口，阻塞等待 EagerGraphExecutor 完成计算 |

### 2.2 关键数据结构

```cpp
class GraphConstructor {
    // 每个 Operand 的缓存：名称 + API Tensor 引用计数
    std::unordered_map<const Operand*, OperandCache> mOperandCaches;

    // 拥有 EagerGraphExecutor 实例
    std::unique_ptr<EagerGraphExecutor> mEagerGraphExecutor;

    // EagerGraphExecutor 的消息队列引用（非拥有）
    EGEMessageQueue& mEGEMessageQueue;
};
```

### 2.3 AddOperator 流程

`AddOperator` 是 GraphConstructor 最核心的方法，每次用户调用算子时触发：

```
1. 校验所有输入 Tensor 属于同一个 Graph
2. 调用 OperatorFactory::NewOperatorOrThrow(OpParam, InputOperands) 创建 Operator
3. 调用 EagerGraphExecutor::CheckSupportOrThrow() 校验设备兼容性
4. SendOperatorToExecutor():
   a. 为每个输出 Operand 初始化 OperandCache
   b. 封装为 AddOperatorEEMsg，推入 EGEMessageQueue
5. 用输出 Operand + Graph 引用构造 api::cpp::Tensor 返回给 Python
```

关键代码（`graph_constructor.cc`）：

```cpp
api::cpp::TensorArray GraphConstructor::AddOperator(Graph& graph, std::unique_ptr<OpParam> opParamPtr,
                                                    const api::cpp::TensorArray& inputs) {
    // 1. Check same graph
    // 2. Construct operator via OperatorFactory
    std::unique_ptr<Operator> op = OperatorFactory::GetSingleton().NewOperatorOrThrow(...);
    // 3. Check device support
    mEagerGraphExecutor->CheckSupportOrThrow(*op);
    // 4. Send to executor (async message)
    SendOperatorToExecutor(std::move(op));
    // 5. Return output api::cpp::Tensor
    return tensorArray;
}
```

### 2.4 消息类型

GraphConstructor 通过 `EGEMessageQueue` 向 EagerGraphExecutor 发送以下消息：

| 消息类型 | 基类 | 用途 | 是否阻塞 Producer |
|---|---|---|---|
| `AddOperatorEEMsg` | `EGEMessage` | 投递新 Operator | 否（异步） |
| `ApiTensorNoHoldEEMsg` | `EGEMessage` | 通知 Operand 不再被 API Tensor 引用 | 否（异步） |
| `WaitAllFinishEEMsg` | `FutureEGEMessage<void>` | 等待所有已提交算子执行完毕 | **是**（同步） |
| `GetTensorEEMsg` | `FutureEGEMessage<GetTensorResult>` | 获取 Tensor 的实际值 | **是**（同步） |
| `SetGraphNameEEMsg` | `EGEMessage` | 设置图名称 | 否（异步） |
| `SetOperandNameEEMsg` | `EGEMessage` | 设置 Operand 名称 | 否（异步） |

> **关键设计**：异步消息（`EGEMessage`）不阻塞 Producer 线程，Python Client 可以持续构建计算图。同步消息（`FutureEGEMessage`）仅在需要获取 Tensor 值或显式同步时阻塞。

### 2.5 Operand 引用计数

GraphConstructor 通过 `ApiTensorRefCountHolder`（RAII 守卫）管理 Python 侧 Tensor 对 Operand 的引用计数：

- Python Tensor 构造时，`ApiTensorRefCountHolder` 构造 → `ApiTensorRefCountIncrease()`
- Python Tensor 析构时，`ApiTensorRefCountHolder` 析构 → `ApiTensorRefCountDecrease()`
- 当引用计数降为 0 → 发送 `ApiTensorNoHoldEEMsg` 通知 EagerGraphExecutor 可释放该 Operand

这确保 Operand 的物理内存（Blob）仅在没有任何 API Tensor 引用时才会被回收。

---

## 3. EagerGraphExecutor — 图执行器

`EagerGraphExecutor` 是 Single-Controller 架构中的 **Controller** 实现。它在独立的 `AsyncMain` 线程中运行，负责消费消息、管理计算图生命周期、调度算子执行。

源码位置：`dtorch/core/graph/eager_graph_executor.h` / `.cc`

### 3.1 核心职责

| 职责 | 说明 |
|---|---|
| **消息消费** | 从 `EGEMessageQueue` 批量拉取消息（每次最多 100 条），处理 Operator 添加、Operand 释放、同步等待等请求 |
| **计算图管理** | 将 Operator 和 Operand 插入 `LogicalGraph`，执行后删除 |
| **拓扑执行** | 通过 `GraphTraversalSequence` 维护算子拓扑序，按序执行 |
| **远端广播** | 持有 `RemoteRunnerPublisher`（PUB socket），将算子序列化后广播给所有远端子进程 |
| **Runner 管理** | 持有 `std::vector<std::unique_ptr<NodeRunnerBase>>`，NodeRunnerBase 持有当前节点的 NodeRunner（即 `PerDeviceThreadNodeRunner` 或 `PerDeviceProcessNodeRunner`） |
| **同步响应** | 处理 `GetTensor` / `Wait` 等阻塞请求，在算子执行完毕后响应 |
| **远端就绪等待** | 持有 `RemoteRunnerPuller`（PULL socket），等待所有远端 Runner 就绪后再继续（在多进程/多节点模式下） |

### 3.2 关键数据结构

```cpp
class EagerGraphExecutor {
    // 异步主循环线程
    std::thread mAsyncThread;
    std::atomic_bool mGetDestroySignal;

    // 消息队列
    EGEMessageQueue mEGEMessageQueue;

    // 本批次的阻塞消息集合
    BlockEGEMessageCollections mNewAddBlockMessage;

    // 本批次新增的 Operator 序列（按添加顺序）
    std::vector<const Operator*> mNewAddOperatorSequence;

    // 本批次不再被引用的 Operand
    std::vector<const Operand*> mNewAddNoHoldOperand;

    // 计算图
    LogicalGraph mLogicalGraph;

    // Runner 列表（单机场景只有 main node 的 Runner）
    std::vector<std::unique_ptr<NodeRunnerBase>> mNodeRunners;
    std::unique_ptr<external::zmq::RemoteRunnerPublisher> mRemoteRunnerPublisher;
    std::unique_ptr<external::zmq::RemoteRunnerPuller> mRemoteRunnerPuller;
};
```

### 3.3 AsyncMain 主循环

`AsyncMain` 是 EagerGraphExecutor 的核心，运行在独立线程中，循环执行以下步骤（`eager_graph_executor.cc:165-202`）：

```
while (true) {
    1. GetEagerGraphExecutorMessage()
       → 从消息队列批量拉取消息，填充 mNewAddOperatorSequence /
         mNewAddNoHoldOperand / mNewAddBlockMessage
       → 每条消息通过虚函数 ProcessEGEMessage() 回调到 EagerGraphExecutor 的对应方法

    2. 构建 GraphTraversalSequence
       → 将 mNewAddOperatorSequence 按拓扑序遍历，生成执行序列

    3. RespondWaitMessageEarly()
       → 如果用户仅请求 Wait 但没有待执行算子，提前响应

    4. ExecuteOperatorsAndNoHoldOperands(traversalSeq, noHoldOperands)
       → 从 LogicalGraph 取出 Operator shared_ptr
       → 调用所有 Runner 的 Execute(ops, noHoldOperands)
       → 删除已执行的 Operator 和不再引用的 Operand

    5. RespondWaitMessage()
       → 如果有 GetTensor 请求 → 调用 Runner::GetTorchTensor() 获取值
       → 如果有 Wait 请求 → 调用 Runner::Sync() 同步所有流
       → 循环等待直到 main thread 不再阻塞
}
```

### 3.4 当前节点的 Runner 初始化

在构造函数中，EagerGraphExecutor 通过 `InitCurrentNode()` 为当前节点创建 Runner，根据 `GraphOption::perDevicePerProcess` 决定 Runner 类型：

```cpp
// eager_graph_executor.cc InitCurrentNode()
if (graphOption.perDevicePerProcess.value()) {
    mainNodeRunner = std::make_unique<PerDeviceProcessNodeRunner>(...);
} else {
    mainNodeRunner = std::make_unique<PerDeviceThreadNodeRunner>(...);
}
```

- `perDevicePerProcess = false`：使用 `PerDeviceThreadNodeRunner`（当前进程内线程），**无需**远端等待。
- `perDevicePerProcess = true`：使用 `PerDeviceProcessNodeRunner`（子进程），按设备逐个创建远端 Runner，通过 `RemoteRunnerPuller` 收集每个 device 的就绪通知，确保所有远端 Runner 就绪后才继续流程。

> 本文档仅覆盖 `perDevicePerProcess = false`（默认）的线程模式，即使用 `PerDeviceThreadNodeRunner`。

### 3.5 算子执行与资源释放

`ExecuteOperatorsAndNoHoldOperands`（`eager_graph_executor.cc:218-241`）：

```cpp
void EagerGraphExecutor::ExecuteOperatorsAndNoHoldOperands(...) {
    // 从 LogicalGraph 取出 shared_ptr<Operator>（转移所有权）
    std::vector<std::shared_ptr<Operator>> ops;
    for (auto op : traversalSeq.ToVec()) {
        ops.push_back(mLogicalGraph.GetOperator(op));
        mLogicalGraph.DeleteOperator(op);  // 从图中移除
    }

    // 分发给所有 Runner 执行
    for (auto& runner : mNodeRunners) {
        runner->Execute(ops, noHoldOperands);
    }

    // 释放不再被引用的 Operand
    for (auto operand : noHoldOperands) {
        mLogicalGraph.DeleteOperand(operand);
    }
}
```

每个 Operator **只执行一次**，执行完毕后立即从 LogicalGraph 中删除。这种 "eager" 策略确保了显存占用的最小化。

### 3.6 BlockEGEMessageCollections

`BlockEGEMessageCollections` 聚合了需要阻塞 Producer 线程的请求：

```cpp
struct BlockEGEMessageCollections {
    bool mainThreadIsWait;          // Producer 线程是否正在阻塞等待
    bool waitAllFinish;             // 是否等待所有算子完成
    std::promise<void> waitAllFinishPromise;
    const Operand* getThisTensor;   // 需要获取值的 Operand
    std::promise<GetTensorResult> getTensorPromise;
};
```

当 `GetTensor` 或 `Wait` 请求到达时，`mainThreadIsWait` 被设为 `true`。AsyncMain 在执行完当前批次算子后，检查该标志并调用相应的 Runner 方法获取结果，然后通过 `std::promise` 唤醒阻塞的 Producer 线程。

---

## 4. Runner 执行层

### 4.1 RunnerBase — 抽象基类

`RunnerBase` 定义了所有 Runner 的统一接口（`dtorch/core/runner/runner_base.h`）：

```cpp
class RunnerBase {
public:
    virtual void Execute(
        const std::vector<std::shared_ptr<Operator>>& ops,
        const std::vector<const Operand*>& noHoldOperands) = 0;

    virtual std::shared_ptr<torch::Tensor> GetTorchTensor(const Operand* operand) = 0;

    virtual void Sync() = 0;
};
```

三个纯虚函数分别对应执行、取值、同步三类操作。

### 4.2 PerDeviceThreadNodeRunner — 单节点线程 Runner

`PerDeviceThreadNodeRunner` 是 `RunnerBase` 的线程级实现，用于单机场景。它是一个**薄封装层**，将所有调用直接委托给内部的 `NaiveRunner`。

源码位置：`dtorch/core/runner/per_device_thread_node_runner.h`

```cpp
class PerDeviceThreadNodeRunner : public RunnerBase {
public:
    PerDeviceThreadNodeRunner(const GraphOption& graphOption,
                              const RunnerSupportedDevices& supportedDevices)
        : mNaiveRunner(graphOption, supportedDevices,
                       communication::TensorStoreConfig(
                           communication::TensorStoreType::kMemory)) {}

    void Execute(...) override { mNaiveRunner.Execute(ops, noHoldOperands); }
    std::shared_ptr<torch::Tensor> GetTorchTensor(...) override {
        DDebugAssert(!operand->IsDistributed());
        return mNaiveRunner.GetTorchTensor(operand);
    }
    void Sync() override { mNaiveRunner.Sync(); }

private:
    NaiveRunner mNaiveRunner;
};
```

关键点：
- 使用 `TensorStoreType::kMemory` — 多个 Kernel 间通过**内存**共享中间结果（而非共享内存文件）。
- `GetTorchTensor` 断言 Operand 不是分布式的 — 因为从单个 Runner 只能获取本地非分布式 Tensor。

### 4.3 NaiveRunner — 核心执行引擎

`NaiveRunner` 是算子执行的**核心引擎**，负责将 Operator 转换为 Kernel、管理 Blob 生命周期、调度 KernelStream 执行。

源码位置：`dtorch/core/runner/naive_runner.h` / `.cc`

#### 4.3.1 核心职责

| 职责 | 说明 |
|---|---|
| **Operator → Kernel 转换** | 遍历每个 Operator，调用 `Kernel::CreateKernel()` 为每个设备/流创建 Kernel 实例 |
| **Blob 管理** | 维护 `Operand → AllDeviceBlobs` 映射，管理物理显存的分配与释放 |
| **KernelStream 管理** | 通过 `KernelStreamManager` 管理 CUDA Stream / CPU Stream 的创建与复用 |
| **ThreadGroup 管理** | 通过 `ThreadGroupManager` 管理集合通信所需的线程组 |
| **同步与取值** | `Sync()` 同步所有 Stream；`GetTorchTensor()` 从 Blob 中取出 `torch::Tensor` |

#### 4.3.2 关键数据结构

```cpp
class NaiveRunner {
    RunnerSupportedDevices mSupportedDevices;            // 支持的设备列表
    std::unordered_map<const Operand*, AllDeviceBlobs>
        mOperandToBlobs;                                 // Operand → 各设备的 Blob 映射
    std::shared_ptr<communication::ThreadGroupManager>
        mThreadGroupManager;                             // 集合通信线程组
    KernelStreamManager mStreamManager;                  // KernelStream 管理器
    const communication::TensorStoreConfig mStoreConfig; // Tensor 存储配置
};
```

#### 4.3.3 Execute 执行流程

`NaiveRunner::Execute`（`naive_runner.cc:40-60`）：

```
1. 遍历所有 Operator，调用 CreateKernelForOperator() 创建 Kernel
2. 遍历所有 noHoldOperands，调用 DeleteOperand() 释放 Blob
3. 遍历所有 Kernel，调用 kernel->GetStream().LaunchKernel() 发射到流中执行
```

#### 4.3.4 CreateKernelForOperator — Kernel 创建

`CreateKernelForOperator`（`naive_runner.cc:62-125`）是执行流程中最关键的方法：

```
1. 获取 OperatorAssignInfo，确定需要创建多少个 Kernel (NumKernelForThisOp)
2. 如果 Kernel 数量 > 1，创建 TensorStoreCreateInfo 用于 Kernel 间通信
3. 收集输入 Blob：从 mOperandToBlobs 查找输入 Operand 对应的 Blob
4. 分配输出 Blob：为输出 Operand 的每个 DeviceMesh 设备创建新 Blob
5. 遍历支持的 StreamKey，为每个 (globalDevice, localDevice, streamType) 组合：
   a. 通过 mStreamManager.GetStream() 获取或创建 KernelStream
   b. 构造 KernelCreateCtx（包含输入/输出 Blob、Stream、ThreadGroup 等）
   c. 调用 Kernel::CreateKernel(ctx) 创建 Kernel 实例
```

**多 Kernel 场景**：当一个 Operator 涉及多个设备时需要创建多个 Kernel（每个设备一个），此时 `TensorStoreCreateInfo` 被创建，用于协调多个 Kernel 间的数据同步。

#### 4.3.5 Blob 生命周期

```
创建：Operator 首次产生某个 Operand 时
      → NaiveRunner::NewBlob() 为每个全局设备 ID 创建空 Blob
      → Kernel 执行时 Blob 内部创建 torch::Tensor

读取：后续 Operator 以该 Operand 为输入时
      → 从 mOperandToBlobs 查找已有 Blob

释放：Python 侧 API Tensor 引用计数归零
      → EagerGraphExecutor 收到 ApiTensorNoHoldEEMsg
      → NaiveRunner::DeleteOperand() 从 mOperandToBlobs 移除
```

---

## 5. 完整执行流程

以下以 `c = a + b` 为例，追踪一次加法运算经过四层组件的完整路径：

```
Python: c = a + b
    │
    ▼
GraphConstructor::AddOperator()
    ├── OperatorFactory::NewOperatorOrThrow("add", {a.operand, b.operand})
    │   └── 返回 unique_ptr<Operator>，包含输出 Operand c_operand
    ├── EagerGraphExecutor::CheckSupportOrThrow(*op)
    ├── SendOperatorToExecutor()
    │   ├── InitOperandCache(c_operand)  // 初始化引用计数
    │   └── EGEMessageQueue::PushMessage(AddOperatorEEMsg{op})
    └── 返回 api::cpp::Tensor(c_operand)
    │
    ▼  (异步 — Producer 线程继续，不等待)
    │
EagerGraphExecutor::AsyncMain()  // 消费者线程
    ├── GetEagerGraphExecutorMessage()
    │   └── 处理 AddOperatorEEMsg → AddOperator(op)
    │       ├── LogicalGraph::AddOperand(output_operands)
    │       ├── mNewAddOperatorSequence.push_back(op)
    │       └── LogicalGraph::AddOperator(op)
    │
    ├── GraphTraversalSequence::FromVec(mNewAddOperatorSequence)
    │
    ├── ExecuteOperatorsAndNoHoldOperands(traversalSeq, noHoldOperands)
    │   └── for each runner: runner->Execute(ops, noHoldOperands)
    │       │
    │       ▼
    │   PerDeviceThreadNodeRunner::Execute(ops, noHoldOperands)
    │       └── mNaiveRunner.Execute(ops, noHoldOperands)
    │           │
    │           ▼
    │       NaiveRunner::Execute()
    │           ├── CreateKernelForOperator(add_op)
    │           │   ├── 查找输入 Blob: a.blob, b.blob
    │           │   ├── 分配输出 Blob: c.blob (NewBlob for each device)
    │           │   ├── mStreamManager.GetStream(device, ...)
    │           │   └── Kernel::CreateKernel(ctx)
    │           │       └── 返回 CudaAddKernel / CpuAddKernel
    │           │
    │           └── for each kernel:
    │               kernel->GetStream().LaunchKernel(kernel)
    │               └── 发射到 CUDA Stream 或 CPU 线程执行
    │
    └── LogicalGraph::DeleteOperator(op)  // 执行后立即清理
```

### 同步路径（仅在获取 Tensor 值时触发）

```
Python: value = c.item()
    │
    ▼
GraphConstructor::GetSharedPtrTensor(c_operand)
    └── EGEMessageQueue::PushMessageAndGetResult(GetTensorEEMsg)
        │  (Producer 线程阻塞在 future.get())
        │
        ▼
EagerGraphExecutor::AsyncMain()
    ├── (执行完当前批次所有算子)
    ├── RespondGetTensor()
    │   └── mNodeRunners[0]->GetTorchTensor(c_operand)
    │       └── NaiveRunner::GetTorchTensor()
    │           ├── mStreamManager.Sync()  // 等待所有 CUDA Stream 完成
    │           └── return mOperandToBlobs[operand][deviceId].GetTensor()
    └── promise.set_value(tensor) → 唤醒 Producer 线程
```

---

## 6. 源码索引

| 组件 | 头文件 | 实现文件 |
|---|---|---|
| `GraphConstructor` | `dtorch/core/graph/graph_constructor.h` | `dtorch/core/graph/graph_constructor.cc` |
| `EagerGraphExecutor` | `dtorch/core/graph/eager_graph_executor.h` | `dtorch/core/graph/eager_graph_executor.cc` |
| `EGEMessageQueue` / `EGEMessage` | `dtorch/core/graph/eager_graph_executor_message.h` | — |
| 消息实现（`AddOperatorEEMsg` 等） | `dtorch/core/graph/eager_graph_executor_message_imp.h` | — |
| `GraphTraversalSequence` | `dtorch/core/graph/graph_traversal_sequence.h` | — |
| `LogicalGraph` | `dtorch/core/graph/logical_graph.h` | — |
| `RunnerBase` | `dtorch/core/runner/runner_base.h` | — |
| `PerDeviceThreadNodeRunner` | `dtorch/core/runner/per_device_thread_node_runner.h` | — |
| `NaiveRunner` | `dtorch/core/runner/naive_runner.h` | `dtorch/core/runner/naive_runner.cc` |
| `RunnerSupportedDevices` | `dtorch/core/runner/runner_supported_devices.h` | — |
| `KernelStreamManager` | `dtorch/core/kernel_stream/kernel_stream_manager.h` | — |
| `GraphOption` | `dtorch/api/cpp/graph.h` | — |
