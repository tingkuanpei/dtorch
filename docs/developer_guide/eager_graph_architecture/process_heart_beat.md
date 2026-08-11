# 进程心跳机制 (Process Heart Beat)

DTorch 在单机多进程场景下，使用基于 **gRPC** 的心跳机制监控主进程（Controller）与 GPU 子进程（Worker）之间的存活状态。任一进程异常退出时，对方能及时检测并执行优雅关闭，避免资源泄漏和死锁。

---

## 1. 为什么需要心跳机制

在单机多进程模式下，Controller 与 Worker 之间通过 ZMQ 进行算子分发（数据面），但 ZMQ 本身不提供连接健康检测。需要独立的控制面健康检查来覆盖以下场景：

| 场景 | 检测方 | 处理方式 |
|---|---|---|
| Worker 进程崩溃/卡死 | Main 进程 | 退出整个集群，避免挂起的 ZMQ 连接导致 Controller 阻塞 |
| Main 进程崩溃 | Worker 进程 | Worker 检测到 Main 不可达，设置退出标志并优雅退出 |
| 正常关闭 | 双向 | Main 通知所有 Worker 停止轮询；Worker 从 Main 取消注册 |

心跳机制与算子通信分离，使用 gRPC（而非 ZMQ）作为传输协议，利用其内置的重试策略和超时控制。

---

## 2. 源码位置

| 文件 | 内容 |
|---|---|
| `dtorch/core/distributed/process_heart_beat.h` | 心跳类体系：`ProcessHeartBeatBase` / `MainProcessHeartBeat` / `WorkerProcessHeartBeat` |
| `dtorch/core/distributed/process_heart_beat.cc` | 心跳类实现 |
| `dtorch/external/rpc/heart_beat_interface.h` | gRPC 封装：`HeartBeatServer` / `HeartBeatClient` |
| `dtorch/external/rpc/heart_beat_interface.cc` | gRPC 服务实现与客户端调用 |
| `dtorch/external/rpc/proto/heart_beat.proto` | gRPC 协议定义 (protobuf) |
| `dtorch/core/runner/remote_runner_in_process.cc` | Worker 子进程启动流程，集成 `WorkerProcessHeartBeat` |

---

## 3. 类体系

```
ProcessHeartBeatBase                  (基类：公共轮询基础设施)
├── MainProcessHeartBeat              (Main 进程：监控所有 Worker 心跳)
└── WorkerProcessHeartBeat            (Worker 进程：监控 Main 心跳)
```

### 3.1 基类: `ProcessHeartBeatBase`

提供所有心跳实现共享的基础设施：

```cpp
class ProcessHeartBeatBase {
protected:
    std::mutex mMutex;                  // 保护并发访问
    std::condition_variable mCv;        // 条件变量，用于定时唤醒 and 通知停止
    std::atomic_bool mStopPoll;         // 停止轮询标志（原子变量，跨线程可见）
};
```

核心方法：

| 方法 | 说明 |
|---|---|
| `NotifyStopPoll()` | 设置 `mStopPoll = true`，通过条件变量唤醒轮询线程使其退出 |
| `WaitForStopPoll(timeout = 4000ms)` | 轮询线程调用，等待超时或收到停止通知。既是休眠手段（每 4s 一轮），也是退出检测点 |
| `RegisterWorkerProcess(address)` | 纯虚函数 — 子类实现注册逻辑 |
| `UnregisterWorkerProcess(address)` | 纯虚函数 — 子类实现取消注册逻辑 |

**设计要点**：轮询线程不是持续高频循环，而是 `WaitForStopPoll` 等待 4 秒超时，每次超时执行一次心跳检查。这样既保证了及早故障发现（4s 内），又避免了 CPU 空转。

---

### 3.2 MainProcessHeartBeat

**职责**：运行在 Main 进程（Controller），监控所有已注册 Worker 进程的健康状态。Main 进程也启动一个 gRPC Server，供 Worker 回调注册/取消注册/心跳查询。

```
┌──────────────────────────────────────────────────────────────┐
│  MainProcessHeartBeat (Main 进程)                             │
│                                                              │
│  ┌─────────────────────┐    ┌──────────────────────────────┐ │
│  │ HeartBeatServer     │    │ PollWorkerHeartBeatAsyncMain │ │
│  │ (gRPC, mainAddress) │    │ (后台线程)                    │ │
│  └────────┬────────────┘    └────────────┬─────────────────┘ │
│           │ gRPC 服务                    │ 每 4s 轮询           │
│           ▼                              ▼                     │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  mWorkerHeartBeatClients: unordered_map<addr, Client>  │    │
│  │  (每个已注册 Worker 对应一个 HeartBeatClient)          │    │
│  └───────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### 构造过程

```cpp
MainProcessHeartBeat(const std::string& mainAddress)
    : mMainAddress(mainAddress)
    , mMainHeartBeatServer(mainAddress, *this)   // 创建 gRPC Server，传入自身作为回调
    , mWorkerHeartBeatClients()                  // 初始为空
{
    mPollMainHeartBeatThread = std::thread(&MainProcessHeartBeat::PollWorkerHeartBeatAsyncMain, this);
}
```

1. 创建 `HeartBeatServer` 在 `mainAddress` 上监听 — 提供 `IsBeat` / `NotifyStopPoll` / `RegisterWorkerProcess` / `UnregisterWorkerProcess` 四个 RPC 端点
2. 启动后台轮询线程 `PollWorkerHeartBeatAsyncMain`

#### 轮询逻辑 (`PollWorkerHeartBeatAsyncMain`)

```
while (!mStopPoll):
    for each workerClient in mWorkerHeartBeatClients:
        if !workerClient.IsBeat():        // gRPC 调用，检查 Worker 是否存活
            LogError + StopWorkerPoll()   // 通知所有 Worker 停止轮询
            std::exit(0)                  // 退出 Main 进程
    WaitForStopPoll(4000)                 // 等待 4s 或直到被通知停止
StopWorkerPoll()                          // 退出前通知所有 Worker
```

- **轮询周期**：4 秒（`WaitForStopPoll` 默认超时）
- **失败处理**：任一 Worker 不响应 → 记录错误 → 通知所有 Worker 停止 → Main 进程退出
- **退出清理**：析构时调用 `NotifyStopPoll()` 唤醒轮询线程，join 线程，然后 `StopWorkerPoll()` 通知所有 Worker 客户端停止轮询

#### 注册 / 取消注册

```cpp
void RegisterWorkerProcess(const std::string& workerAddress) {
    // 由 gRPC 回调触发（Worker 主动注册）
    mWorkerHeartBeatClients.emplace(workerAddress, HeartBeatClient(mainAddress));
}

void UnregisterWorkerProcess(const std::string& workerAddress) {
    // 由 gRPC 回调触发（Worker 退出时取消注册）
    mWorkerHeartBeatClients.erase(workerAddress);
}
```

这两个方法由 `HeartBeatServer` 接收到 Worker 的 gRPC 请求后回调触发，运行在 gRPC 的服务线程中而非轮询线程，因此使用 `mMutex` 保护 `mWorkerHeartBeatClients`。

---

### 3.3 WorkerProcessHeartBeat

**职责**：运行在每个 Worker（GPU 子进程）中，监控 Main 进程的健康状态。Worker 进程启动自己的 gRPC Server 供 Main 回调。

```
┌─────────────────────────────────────────────────────────────┐
│  WorkerProcessHeartBeat (Worker 进程)                        │
│                                                             │
│  ┌──────────────────────┐    ┌────────────────────────────┐ │
│  │ HeartBeatServer      │    │ PollMainHeartBeatAsyncMain │ │
│  │ (gRPC, workerAddr)   │    │ (后台线程)                  │ │
│  └──────────────────────┘    └────────────┬───────────────┘ │
│                                           │ 每 4s 轮询       │
│                                           ▼                  │
│                              ┌────────────────────────────┐ │
│                              │ mMainHeartBeatClient       │ │
│                              │ (HeartBeatClient,          │ │
│                              │  连接 Main 的 gRPC Server) │ │
│                              └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### 构造过程

```cpp
WorkerProcessHeartBeat(const std::string& mainAddress, const std::string& thisWorkerAddress,
                       std::function<void()> onExit = nullptr)
    : mMainAddress(mainAddress)
    , mThisWorkerAddress(thisWorkerAddress)
    , mWorkerHeartBeatServer(thisWorkerAddress, *this)   // 创建自身 gRPC Server
    , mMainHeartBeatClient(mainAddress)                  // 创建到 Main 的 gRPC Client
    , mOnExit(std::move(onExit))                        // 退出回调
{
    // 立即向 Main 进程注册
    if (!mMainHeartBeatClient.RegisterWorker(thisWorkerAddress)) {
        throw std::runtime_error("Register WorkerProcessHeartBeat to main process failed");
    }
    mPollMainHeartBeatThread = std::thread(&WorkerProcessHeartBeat::PollMainHeartBeatAsyncMain, this);
}
```

1. 创建自身 `HeartBeatServer` 用于接收 Main 的回调（`IsBeat` 查询、`NotifyStopPoll` 通知等）
2. 创建 `HeartBeatClient` 连接到 Main 进程的 gRPC Server
3. **立即调用 `RegisterWorker`** 向 Main 注册自己的地址 — 失败时抛异常，阻止 Worker 启动
4. 存储可选的 `onExit` 回调 — 轮询线程退出时调用，用于通知主线程退出
4. 启动后台轮询线程监控 Main 进程

#### 轮询逻辑 (`PollMainHeartBeatAsyncMain`)

```
while (!mStopPoll):
    if !mMainHeartBeatClient.IsBeat():   // gRPC 调用 Main 进程的心跳接口
        LogError
        if (mOnExit) mOnExit()           // 调用退出回调
        break                            // 退出轮询循环
    WaitForStopPoll(4000)                // 等待 4s
```

- **轮询周期**：4 秒
- **失败处理**：Main 不响应 → 调用构造函数传入的 `onExit` 回调 → 退出轮询线程
- 回调由 `BackgroundProcessExecMain` 设置为调用 `processSync.NotifyExit()`，唤醒共享内存条件变量，使得 `processSync.WaitForExit()` 返回，Worker 进程正常退出

Worker **不支持** `RegisterWorkerProcess` / `UnregisterWorkerProcess` — 这两个方法由 Main 端实现，Worker 端调用会触发 `DUnsupportedImpl()`。

#### 析构过程

```cpp
~WorkerProcessHeartBeat() {
    mMainHeartBeatClient.UnregisterWorker(mThisWorkerAddress);  // 从 Main 取消注册
    NotifyStopPoll();                                           // 通知轮询线程停止
    if (mPollMainHeartBeatThread.joinable()) {
        mPollMainHeartBeatThread.join();                        // 等待线程退出
    }
}
```

---

## 4. gRPC 接口定义

心跳机制基于 `heart_beat.HeartBeat` gRPC 服务：

```protobuf
service HeartBeat {
    rpc IsBeat(HeartBeatRequest) returns (HeartBeatReply) {}
    rpc NotifyStopPoll(HeartBeatRequest) returns (HeartBeatReply) {}
    rpc RegisterWorkerProcess(RegisterWorkerProcessRequest) returns (RegisterWorkerProcessReply) {}
    rpc UnregisterWorkerProcess(RegisterWorkerProcessRequest) returns (RegisterWorkerProcessReply) {}
}
```

| RPC | 调用方向 | 说明 |
|---|---|---|
| `IsBeat` | 双向 | 健康检查 — 收到请求并返回 OK 即表示进程存活 |
| `NotifyStopPoll` | 双向 | 通知对方停止心跳轮询（用于关闭流程） |
| `RegisterWorkerProcess` | Worker → Main | Worker 启动时向 Main 注册自己的地址 |
| `UnregisterWorkerProcess` | Worker → Main | Worker 退出时从 Main 取消注册 |

### 4.1 gRPC Server 端 (`HeartBeatServer`)

继承 `RpcServer`，内部创建 `HeartBeatServiceImpl` 处理四个 RPC：

- `IsBeat` / `NotifyStopPoll` / `RegisterWorkerProcess` / `UnregisterWorkerProcess` 均转发到 `ProcessHeartBeatBase&` 引用的对应方法
- 所有 RPC 返回 `grpc::Status::OK`（失败由 gRPC 框架层面处理）

### 4.2 gRPC Client 端 (`HeartBeatClient`)

封装到对端 gRPC Server 的调用，使用带重试策略的 Channel：

```cpp
// 重试策略 (JSON 配置)
{
    "maxAttempts": 4,                              // 最多重试 4 次
    "initialBackoff": "1s",                        // 初始退避 1s
    "maxBackoff": "5s",                            // 最大退避 5s
    "backoffMultiplier": 2.0,                      // 退避倍增因子
    "retryableStatusCodes": ["UNAVAILABLE"]        // 仅 UNAVAILABLE 状态码触发重试
}
```

重试策略确保临时网络抖动不会导致误判进程死亡。最多 4 次尝试累计耗时约 15s（1 + 2 + 4 + 5），远超轮询间隔（4s），但大多数情况下首次调用即成功，不影响正常心跳。

### 4.3 心跳交互模型

```
  Main Process                          Worker Process
  ────────────                          ──────────────
  HeartBeatServer ◄──── IsBeat ──────── HeartBeatClient (Worker 侧)
  (mainAddress)                        (连接 mainAddress)

  HeartBeatClient  ────── IsBeat ──────► HeartBeatServer  (Main 侧)
  (连接 workerAddr)                     (workerAddress)
```

每对方均有**一对 Server + Client**，形成双向独立的心跳通道。Main 侧的 Client 和 Worker 侧的 Server 配对，Worker 侧的 Client 和 Main 侧的 Server 配对。

---

## 5. 与多进程启动流程的集成

心跳机制集成在 `RemoteRunnerInProcessLauncher` 的启动流程中：

### 5.1 Main 进程侧

在 `MainNode` 构造时创建 `MainProcessHeartBeat`：

```cpp
// dtorch/core/distributed/main_node.cc
MainNode::MainNode(const std::string& mainNodeAddress) {
    std::string mainProcessHeartBeatAddress = Cluster::GetValidNodeAddress();
    mMainProcessHeartBeat = std::make_unique<MainProcessHeartBeat>(mainProcessHeartBeatAddress);
    // MainProcessHeartBeat 构造后立即启动心跳 Server 和轮询线程
}
```

Main 进程启动 Worker 子进程时，通过共享内存将 `mainProcessHeartBeatAddress` 传递给子进程：

```cpp
// dtorch/core/runner/remote_runner_in_process.cc (Launcher 端)
const std::string mainProcessHeartBeatAddress =
    Cluster::GetSingleton().GetMainProcessHeartBeatAddress();
boa << mainProcessHeartBeatAddress;  // 序列化到共享内存
```

### 5.2 Worker 子进程侧

子进程启动后创建 `WorkerProcessHeartBeat`，并传入退出回调：

```cpp
// dtorch/core/runner/remote_runner_in_process.cc (BackgroundProcessExecMain)
// Step 1: 从共享内存读取 Main 进程心跳地址
std::string mainProcessHeartBeatAddress;
bia >> mainProcessHeartBeatAddress;

// Step 2: 启动 RemoteRunner
RemoteRunner runner(...);

// Step 3: 通知父进程已启动
processSync.NotifyProcessStarted();

// Step 4: 启动心跳，向 Main 注册。
// 传入 onExit 回调：当心跳检测到 Main 崩溃时，调用 processSync.NotifyExit()
// 唤醒共享内存条件变量，使 WaitForExit() 返回，Worker 进程正常退出。
std::string workerHeartBeatAddress = external::rpc::GetRandomUdsAddress();
auto workerProcessHeartBeat = std::make_unique<WorkerProcessHeartBeat>(
    mainProcessHeartBeatAddress, workerHeartBeatAddress,
    [&processSync]() { processSync.NotifyExit(); });

// Step 5: 等待退出信号（父进程通知 或 心跳回调）
processSync.WaitForExit();
```

---

## 6. 完整生命周期时序

```
Main 进程                            Worker 进程
────────                             ──────────
MainProcessHeartBeat 构造
  ├─ HeartBeatServer 启动
  └─ 轮询线程启动 (无 Worker，空循环)
                                     fork + exec → 子进程启动
                                                    │
                                      WorkerProcessHeartBeat 构造
                                        ├─ HeartBeatServer 启动 (workerAddr)
                                        ├─ HeartBeatClient 连接 mainAddress
                                        ├─ RegisterWorker RPC ──────────► Main 收到注册，添加到 mWorkerHeartBeatClients
                                        └─ 轮询线程启动

  ┌── 轮询周期 ───────────────────────────────────────┐
  │  for each Worker:                                  │
  │    IsBeat RPC ──────────────────────────────────►  HeartBeatServer 响应 OK
  │                                                    │
  │                                ┌── 轮询周期 ───────┐
  │                                │  IsBeat RPC ───►  HeartBeatServer 响应 OK
  │                                └──────────────────┘
  └───────────────────────────────────────────────────┘

... 正常运行，每 4s 双向心跳 ...

=== 场景 A: Worker 异常退出 ===
  ┌── 轮询周期 ───────────────────────────────────────┐
  │  IsBeat RPC ───────────✗────►  (Worker 已死)       │
  │  LogError                                          │
  │  StopWorkerPoll():                                 │
  │     NotifyStopPoll RPC → 其他 Workers               │
  │  std::exit(0)                                      │
  └───────────────────────────────────────────────────┘

=== 场景 B: Main 异常退出 ===
                                    ┌── 轮询周期 ───────┐
                                    │  IsBeat RPC ─✗─► (Main 已死)
                                    │  LogError         │
                                    │  mOnExit()        │
                                    │  break            │
                                    └──────────────────┘
                                    processSync.NotifyExit()
                                    WaitForExit() 返回
                                    Worker 进程退出
```

### 正常关闭流程

```
Main 进程析构：                          Worker 进程析构：
  NotifyStopPoll()                          mMainHeartBeatClient.UnregisterWorker()
    → 唤醒轮询线程                          NotifyStopPoll()
  join 轮询线程                              → 唤醒轮询线程
  StopWorkerPoll()                          join 轮询线程
    → 通知所有 Worker StopPoll
```

---

## 7. 设计要点

### 7.1 数据面与控制面分离

| 层级 | 协议 | 用途 |
|---|---|---|
| 数据面 | ZMQ (PUB-SUB / REQ-REP) | Operator 广播、Tensor 取值、同步 |
| 控制面 | gRPC | 心跳健康检查、Worker 注册/取消注册 |

心跳走控制面，不影响算子传输性能。即使 ZMQ 通道因背压暂时阻塞，心跳仍能独立运行，不会产生连锁故障。

### 7.2 快速失败 (Fail-Fast)

- Main 进程检测到 Worker 死亡后立即 `std::exit(0)`，不尝试恢复或重连
- Worker 进程检测到 Main 死亡后调用构造时传入的 `onExit` 回调（通知共享内存条件变量），子进程从 `WaitForExit()` 返回并正常退出
- 不引入复杂的 leader 选举或故障恢复机制 — 多进程管理由外部编排系统（如 Kubernetes）负责

### 7.3 基于注册的 Worker 管理

- Worker 必须在启动后**主动注册**（`RegisterWorker` RPC）才能被 Main 监控
- 构造时注册失败则抛异常，阻止 Worker 启动 — 避免"僵尸"进程
- 析构时主动取消注册 — 清理 Main 侧的 Client 映射

### 7.4 gRPC 重试策略

心跳 Client 配置了自动重试（最多 4 次，退避 1s~5s），避免网络瞬时抖动导致误杀进程。4 秒轮询间隔与重试窗口有重叠，实际检测延迟在 4s 到 ~15s 之间（取决于重试次数）。

### 7.5 Unix Domain Socket 传输

所有 gRPC 通信使用 Unix Domain Socket（UDS），地址格式为 `unix:///tmp/DTorch/...`，由 `GetRandomUdsAddress()` 生成。UDS 在同一台机器内提供极低延迟的 IPC，无需端口分配或网络配置。

---

## 8. 与其他组件的依赖关系

```
MainNode ──owns──► MainProcessHeartBeat
                      ├── HeartBeatServer (继承 RpcServer)
                      └── HeartBeatClient × N (每个 Worker 一个)

Worker Process ──owns──► WorkerProcessHeartBeat
                           ├── HeartBeatServer (继承 RpcServer)
                           └── HeartBeatClient × 1 (连接 Main)
```

在 `Cluster` API 层，`MainNode::GetMainProcessHeartBeatAddress()` 暴露心跳地址，供 `RemoteRunnerInProcessLauncher` 通过共享内存传递给子进程。

参见 [Eager Graph 引擎 (单机多进程)](eager_graph_engine_in_cluster.md) 了解更多多进程架构细节。
