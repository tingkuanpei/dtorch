# ZMQ 通信机制

DTorch 在单机多进程场景下，使用 **ZMQ (ZeroMQ / libzmq)** 实现父进程（Controller）与 GPU 子进程（Worker）之间的高性能 IPC 通信。本文档介绍 ZMQ 层的设计，不展开 Runner 的执行逻辑。
---

## 1. 为什么选择 ZMQ

单机多进程场景下，DTorch 需要解决以下通信需求：

| 需求 | 说明 |
|---|---|
| **一对多广播** | Controller 需要将同一批 Operator 同时发送给所有 GPU 子进程执行 |
| **启动就绪上报** | 每个 GPU 子进程启动后需向 Controller 上报自己负责的设备列表 |
| **进程隔离** | 每个 GPU 运行在独立子进程中，进程间通过 IPC 通信 |

ZMQ 原生支持多种 Socket 模式，且提供异步、无锁的消息队列，非常适合此场景。DTorch 使用 **Unix Domain Socket (IPC)** 作为传输层，所有通信在同一台机器内完成，延迟极低。

ZMQ 层只承担两类通信：**算子广播**（PUB-SUB）与**就绪上报**（PUSH-PULL）。

---

## 2. 源码位置

所有 ZMQ 层代码位于 `dtorch/external/zmq/`：

| 文件 | 内容 |
|---|---|
| `zmq.h` | 基础工具函数：超时、收发包装、IPC 地址生成 |
| `remote_runner_publisher.h` | PUB 端：`RemoteRunnerPublisher` + `PublishMessageIdManager` |
| `remote_runner_subscriber.h` | SUB 端：`RemoteRunnerSubscriber` |
| `remote_runner_pusher.h` | PUSH 端：`RemoteRunnerPusher`（子进程上报就绪） |
| `remote_runner_puller.h` | PULL 端：`RemoteRunnerPuller`（父进程收集就绪） |

测试文件：
| 文件 | 内容 |
|---|---|
| `dtorch/tests/test_zeromq.cc` | ZMQ 基础模式测试（PUB/SUB、PUSH/PULL、进程间通信） |
| `dtorch/tests/test_remote_runner.cc` | 端到端集成测试（Publisher → Subscriber 完整流程） |

---

## 3. 工具函数层 (`zmq.h`)

`dtorch::external::zmq` 命名空间提供了一组轻量级的 ZMQ 操作封装：

### 3.1 超时配置

```cpp
DTORCH_FORCEINLINE int GetZmqTimeoutMilliSecond() {
    return static_cast<int>(core::GlobalOption::GetSingleton().GetZmqTimeoutSecond() * 1000);
}
```

从全局配置读取 ZMQ 超时时间（秒），转换为毫秒。所有 ZMQ Socket 的 `rcvtimeo` / `sndtimeo` 均通过此函数设置。

### 3.2 发送/接收包装

```cpp
// 发送多帧消息，自动校验发送帧数
template <size_t N>
void SendMultipart(::zmq::socket_t& socket, const std::array<::zmq::const_buffer, N>& buffers);

// 接收多帧消息，可选阻塞等待
void RecvMultipart(::zmq::socket_t& socket, std::vector<::zmq::message_t>& buffers,
                   size_t expectedCount = 1, bool waitUntilRecv = false);
```

- `SendMultipart`：封装 `zmq::send_multipart`，发送后断言返回值匹配预期帧数。
- `RecvMultipart`：封装 `zmq::recv_multipart`。当 `waitUntilRecv=true` 时，自旋等待直到数据到达；否则使用 Socket 自身的超时设置。
- `GetMsgAsString`：将 `zmq::message_t` 转为 `std::string`。

### 3.3 IPC 地址生成

```cpp
const std::string GetRandomZmqIpcAddress(size_t length = 16) {
    return "ipc://" + GetTempDirectoryPath() + "/DTorch_ZMQ_IPC_" + GetRandomFileName(length) + ".sock";
}
```

每次调用生成唯一的 Unix Domain Socket 地址，格式为 `ipc://<tmpdir>/DTorch_ZMQ_IPC_<random>.sock`。Publisher 的 PUB 地址与 Puller 的 PULL 地址各自独立，避免地址冲突。

---

## 4. 双 Socket 模式设计

DTorch 的 ZMQ 通信层使用两种 Socket 模式，分别承担「算子广播」与「就绪上报」：

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Parent Process (Controller)                   │
│                                                                      │
│  ┌──────────────────────────────┐    ┌──────────────────────────┐    │
│  │  RemoteRunnerPublisher       │    │  RemoteRunnerPuller      │    │
│  │  (PUB socket, bind)          │    │  (PULL socket, bind)     │    │
│  └──────┬───────────────────────┘    └──────────▲───────────────┘    │
│         │  PUB-SUB                              │  PUSH-PULL         │
│         │  (广播 Execute / Destroy)              │  (收集 devicesReady)│
│         │                                       │                    │
├─────────┼───────────────────────────────────────┼────────────────────┤
│         │          Child Process (×N GPU)        │                    │
│         │                                       │                    │
│  ┌──────▼───────────────────────┐    ┌──────────┴───────────────┐    │
│  │  RemoteRunnerSubscriber      │    │  RemoteRunnerPusher       │    │
│  │  (SUB socket, connect)       │    │  (PUSH socket, connect)   │    │
│  └──────────────────────────────┘    └──────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    RemoteRunner                               │    │
│  │            (持有 NaiveRunner，执行反序列化后的 Operator)        │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.1 PUB-SUB（发布-订阅）— 算子广播

**用途**：Controller 向所有 GPU 子进程广播 `Execute` / `Destroy` 命令。

- **Publisher**（`RemoteRunnerPublisher`，由 `EagerGraphExecutor` 持有）：绑定一个 PUB Socket，广播 `Execute`、`Destroy` 两种消息。
- **Subscriber**（每个子进程的 `RemoteRunner` 内部）：连接一个 SUB Socket，订阅所有消息（`""` 前缀过滤器），非阻塞接收。

**为什么需要 PUB-SUB**：当有 N 个 GPU 子进程时，逐个发送相同的 Execute 命令会产生 N 次序列化开销和 N 次网络往返。PUB-SUB 只需一次序列化和一次 `send_multipart`，所有 Subscriber 同时收到消息。

### 4.2 PUSH-PULL（推-拉）— 启动就绪上报

**用途**：子进程启动后向 Controller 上报自己负责的设备列表。

- **Pusher**（子进程 `RemoteRunnerPusher`）：连接一个 PUSH Socket，在 `AsyncMain` 开头发送一次 `devicesReady` 消息。
- **Puller**（`RemoteRunnerPuller`，由 `EagerGraphExecutor` 持有）：绑定一个 PULL Socket，收集各子进程上报的设备列表。

**为什么需要 PUSH-PULL**：Controller 需要知道每个子进程负责哪些设备，以便正确路由。PUSH-PULL 是 ZMQ 标准的无回应单向汇聚模式，N 个子进程的就绪消息会被 Puller 依次拉取。

> 子进程的「进程已启动」信号另由共享内存条件变量（`SubProcessSync`）通知父进程，与 ZMQ 的 `devicesReady` 互为补充：前者表示进程就绪，后者表示 RemoteRunner 已连上 PUB 并上报设备。

### 4.3 消息顺序保证

`Execute`/`Destroy` 全部走同一个 PUB-SUB 通道，ZMQ 保证同一 SUB Socket 内消息的接收顺序与 PUB 端发送顺序一致（且每条消息携带单调递增的 `messageId`，Subscriber 端会校验）。`devicesReady` 走独立的 PUSH-PULL 通道，仅在启动阶段、任何 `Execute` 之前发送一次，因此不存在跨通道的顺序问题。

---

## 5. PublishMessageIdManager — 消息 ID 管理器

`PublishMessageIdManager` 是一个线程安全的单例，负责为每条 PUB 消息分配单调递增的 ID。每个 Publisher 地址独立维护自己的 ID 序列。

```cpp
class PublishMessageIdManager {
    constexpr static int64_t kInitValue = -1;  // 初始值，首条消息 ID 为 0

    // 获取下一条消息的 ID（递增后返回）
    int64_t GetIdAndIncrement(const std::string& address);

    // 获取当前消息 ID（不递增）
    int64_t GetId(const std::string& address);

    std::mutex mMutex;
    std::unordered_map<std::string, int64_t> mMessageIdMap;  // address → lastId
};
```

**使用方式**：`RemoteRunnerPublisher` 在发送每条 PUB 消息时调用 `GetIdAndIncrement()` 获取新 ID。Subscriber 收到消息后校验 `messageId == 上一次 + 1`，确保无丢失、无乱序。

---

## 6. RemoteRunnerPublisher — 广播发送端

`RemoteRunnerPublisher` 是 PUB-SUB 模式的发布端，由 `EagerGraphExecutor` 创建并持有。

### 6.1 内部结构

```cpp
struct RemoteRunnerPublisher::Impl {
    std::string address;
    ::zmq::context_t ctx;
    ::zmq::socket_t publisher;  // PUB socket, bind 模式
};
```

- `sndhwm = 0`：禁用发送高水位标记，消息队列无上限，避免因 Subscriber 消费慢导致阻塞。
- Socket 绑定到随机生成的 IPC 地址（`publisherAddress`）。

### 6.2 消息类型

| 常量 | 字符串值 | 用途 | 帧数 |
|---|---|---|---|
| `kExecuteStr` | `"publisherExecute"` | 广播 Operator 序列化数据 | 3 |
| `kDestroyStr` | `"publisherDestroy"` | 通知所有子进程退出 | 2 |

### 6.3 消息帧格式

每条 PUB 消息由 2 或 3 帧组成：

```
Frame 0: messageId    (字符串，如 "0", "1", "2", ...)
Frame 1: messageType  (kExecuteStr / kDestroyStr)
Frame 2: payload      (仅 Execute 消息有此帧 — Boost 序列化的 Operator 数据)
```

### 6.4 Execute 消息的序列化

```cpp
void RemoteRunnerPublisher::Execute(ops, noHoldOperands) {
    // 1. 将所有 Operator 转换为 OperatorSerializationPack
    // 2. 将 Operand 指针转换为 uintptr_t
    // 3. 通过 Boost BinaryOArchive 序列化为二进制
    // 4. 获取消息 ID
    // 5. 发送 3 帧消息: [messageId, kExecuteStr, serializedData]
}
```

`OperatorSerializationPack` 包含：输入 Operand 的 `uintptr_t`、输出 Operand 的 `uintptr_t`、算子参数（`OpParam`）、唯一 ID。这使得子进程可以完整重建 Operator 对象。详见 [序列化文档](../eager_graph_architecture/serialization.md)。

### 6.5 析构时的 Destroy 通知

```cpp
RemoteRunnerPublisher::~RemoteRunnerPublisher() { SendDestroy(); }
```

析构时发送 `kDestroyStr` 消息，通知所有子进程的 `RemoteRunner` 退出其 `AsyncMain` 循环。

---

## 7. RemoteRunnerSubscriber — 广播接收端

`RemoteRunnerSubscriber` 在每个 GPU 子进程中运行（由 `RemoteRunner` 持有），连接 Publisher 的 PUB Socket，非阻塞接收 `Execute` / `Destroy` 消息。

### 7.1 内部结构

```cpp
struct RemoteRunnerSubscriber::Impl {
    ::zmq::context_t ctx;
    ::zmq::socket_t subscriber;  // SUB socket, connect 到 publisherAddress
    std::string messageType;     // 最后一条消息的类型
    int64_t messageId;           // 最后一条消息的 ID
    std::string serializedData;  // 最后一条 Execute 消息的序列化数据
};
```

- `rcvhwm = 0`（无接收高水位），订阅所有消息（`""` 前缀），设置 `rcvtimeo`。
- **关键**：构造时通过 ZMQ socket monitor 等待 `ZMQ_EVENT_CONNECTED` 事件后才返回。因为 ZMQ 的 SUB `connect` 是异步的，此举确保 Subscriber 不会错过 Publisher 在连接建立前发出的第一批消息。

### 7.2 Get() —— 非阻塞接收

```cpp
bool Get();  // 有新消息返回 true 并更新内部状态；无消息返回 false（状态不变）
```

收到消息后：

1. 校验帧数为 2（Destroy）或 3（Execute）。
2. 校验 `messageId == 上一次 + 1`（严格递增，否则 `DLogFatal`）。
3. 记录 `messageType`；若为 `kExecuteStr`，额外保存 `serializedData`（第 3 帧）。

调用方（`RemoteRunner::AsyncMain`）据此分发：

- `kExecuteStr` → 反序列化并执行（见下文消息流）。
- `kDestroyStr` → 置位退出标志。

---

## 8. RemoteRunnerPusher / RemoteRunnerPuller — 就绪上报

### 8.1 RemoteRunnerPusher（子进程，PUSH）

```cpp
struct RemoteRunnerPusher::Impl {
    ::zmq::context_t ctx;
    ::zmq::socket_t pusher;  // PUSH socket, connect 到 pushAddress
};
```

`NotifyDevicesReady(devices)`：将设备列表序列化后发送 2 帧消息 `[kDevicesReadyStr, serializedDevices]`，其中 `kDevicesReadyStr = "devicesReady"`。每个子进程在 `RemoteRunner::AsyncMain` 开头调用一次。

### 8.2 RemoteRunnerPuller（父进程，PULL）

```cpp
class RemoteRunnerPuller {
    bool Get();                                       // 非阻塞接收，更新 messageType / readyDevices
    const std::string& GetMessageType() const;        // 期望为 "devicesReady"
    const std::vector<api::cpp::Device>& GetReadyDevice() const;  // 上报的设备列表
};
```

`RemoteRunnerPuller` 由 `EagerGraphExecutor` 持有，绑定 `pushPullAddress`。`EagerGraphExecutor` 在初始化时逐个 `Get()` 收集每个子进程上报的设备列表，据此建立「设备 → 子进程」的路由映射。

---

## 9. 完整消息流

以下展示跨 PUB-SUB 与 PUSH-PULL 两种通道的完整消息流：

```
时间线 →

Parent (EagerGraphExecutor)                  Child (RemoteRunner, per GPU)
═════════════════════════════                ═════════════════════════════

0. 启动握手:
   RemoteRunnerInProcess exec 拉起子进程  ──→  构造 RemoteRunner（连上 PUB / PUSH）
                                                ├── Pusher.NotifyDevicesReady(devs)
   ←── PUSH ["devicesReady", serDevs] ────────   │
   Puller.Get() → 收集设备列表                    └── (同时 processSync.NotifyProcessStarted 经共享内存通知父进程)
   Subscriber 已在 connect 时等待 CONNECTED 事件完成

1. Execute:
   Publisher.Execute(ops)
   → PUB [id=0, "publisherExecute", data] ──→ Subscriber.Get() → ProcessSubscriberExecuteMessage()
                                                              → 反序列化 opPacks → NaiveRunner.Execute()
   (取值：GetTensorOp 作为普通算子随 Execute 广播，在子进程内读取 Blob 并通过
    TensorPromise 异步返回结果，无 ZMQ 往返。)

2. Destroy:
   Publisher.SendDestroy()  (或 ~RemoteRunnerPublisher)
   → PUB [id=N, "publisherDestroy"] ────────→ Subscriber.Get() → mGetDestroySignal = true
                                                              → AsyncMain 退出
```

**关键时序保证**：`Execute`/`Destroy` 共用同一 PUB-SUB 通道，ZMQ 保证同一 SUB Socket 内消息顺序与发送顺序一致，且每条消息携带单调递增的 `messageId` 供 Subscriber 校验。`devicesReady` 在独立 PUSH-PULL 通道上、于任何 `Execute` 之前发送一次，因此不会与算子广播产生乱序。

---

## 10. 源码索引

| 组件 | 头文件 | 实现文件 |
|---|---|---|
| ZMQ 工具函数 | `dtorch/external/zmq/zmq.h` | — |
| `RemoteRunnerPublisher` | `dtorch/external/zmq/remote_runner_publisher.h` | `dtorch/external/zmq/remote_runner_publisher.cc` |
| `PublishMessageIdManager` | `dtorch/external/zmq/remote_runner_publisher.h` | — |
| `RemoteRunnerSubscriber` | `dtorch/external/zmq/remote_runner_subscriber.h` | `dtorch/external/zmq/remote_runner_subscriber.cc` |
| `RemoteRunnerPusher` | `dtorch/external/zmq/remote_runner_pusher.h` | `dtorch/external/zmq/remote_runner_pusher.cc` |
| `RemoteRunnerPuller` | `dtorch/external/zmq/remote_runner_puller.h` | `dtorch/external/zmq/remote_runner_puller.cc` |
| ZMQ 基础测试 | `dtorch/tests/test_zeromq.cc` | — |
| 端到端集成测试 | `dtorch/tests/test_remote_runner.cc` | — |
