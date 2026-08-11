# ZMQ 通信机制

DTorch 在单机多进程场景下，使用 **ZMQ (ZeroMQ / libzmq)** 实现父进程（Controller）与 GPU 子进程（Worker）之间的高性能 IPC 通信。本文档介绍 ZMQ 层的设计，不展开 Runner 的执行逻辑。

---

## 1. 为什么选择 ZMQ

单机多进程场景下，DTorch 需要解决以下通信需求：

| 需求 | 说明 |
|---|---|
| **一对多广播** | Controller 需要将同一批 Operator 同时发送给所有 GPU 子进程执行 |
| **点对点请求-响应** | 需要从某个特定子进程获取 Tensor 值（`GetTorchTensor`）或等待同步完成（`Sync`） |
| **消息顺序保证** | 广播消息和点对点消息之间必须维护严格的先后顺序 |
| **进程隔离** | 每个 GPU 运行在独立子进程中，进程间通过 IPC 通信 |

ZMQ 原生支持多种 Socket 模式，且提供异步、无锁的消息队列，非常适合此场景。DTorch 使用 **Unix Domain Socket (IPC)** 作为传输层，所有通信在同一台机器内完成，延迟极低。

---

## 2. 源码位置

所有 ZMQ 层代码位于 `dtorch/external/zmq/`：

| 文件 | 内容 |
|---|---|
| `zmq.h` | 基础工具函数：超时、收发包装、IPC 地址生成 |
| `remote_runner_publisher.h` | PUB 端：`RemoteRunnerPublisher` + `PublishMessageIdManager` |
| `remote_runner_publisher.cc` | PUB 端实现 |
| `remote_runner_client.h` | REQ 端：`RemoteRunnerClient` |
| `remote_runner_client.cc` | REQ 端实现 |
| `remote_runner_server.h` | SUB + REP 端：`RemoteRunnerServer` |
| `remote_runner_server.cc` | SUB + REP 端实现 |

测试文件：
| 文件 | 内容 |
|---|---|
| `dtorch/tests/test_zeromq.cc` | ZMQ 基础模式测试（REQ/REP、PUB/SUB、进程间通信） |
| `dtorch/tests/test_remote_runner.cc` | 端到端集成测试（Publisher → Server 完整流程） |

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
- `RecvMultipart`：封装 `zmq::recv_multipart`。当 `waitUntilRecv=true` 时，自旋等待直到数据到达（用于 `GetTorchTensor` 等必须获取结果的场景）；否则使用 Socket 自身的超时设置。

### 3.3 IPC 地址生成

```cpp
const std::string GetRandomZmqIpcAddress(size_t length = 16) {
    return "ipc://" + GetTempDirectoryPath() + "/DTorch_ZMQ_IPC_" + GetRandomFileName(length) + ".sock";
}
```

每次调用生成唯一的 Unix Domain Socket 地址，格式为 `ipc://<tmpdir>/DTorch_ZMQ_IPC_<random>.sock`。Publisher 和每个 Client-Server 对各自使用独立的地址，避免地址冲突。

---

## 4. 双 Socket 模式设计

DTorch 的 ZMQ 通信层同时使用两种 Socket 模式，这是由需求的二重性决定的：

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Parent Process                               │
│                                                                      │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐    │
│  │  RemoteRunnerPublisher   │    │  RemoteRunnerClient (×N)     │    │
│  │  (PUB socket, bind)      │    │  (REQ socket, bind)          │    │
│  └──────┬───────────────────┘    └──────────┬───────────────────┘    │
│         │  PUB-SUB                         │  REQ-REP                │
│         │  (广播 Execute/Sync/              │  (点对点 GetTorchTensor │
│         │   GetTorchTensor 通知)             │   / Sync 响应)          │
│         │                                   │                        │
├─────────┼───────────────────────────────────┼────────────────────────┤
│         │          Child Process (×N)       │                        │
│         │                                   │                        │
│  ┌──────┴───────────────────┐    ┌──────────┴───────────────────┐    │
│  │  RemoteRunnerServer      │    │  RemoteRunnerServer           │    │
│  │  (SUB socket, connect)   │    │  (REP socket, connect)        │    │
│  └──────────────────────────┘    └──────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    NaiveRemoteRunner                          │    │
│  │                    (执行 Operator, 管理 Blob)                  │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.1 PUB-SUB（发布-订阅）

**用途**：Controller 向所有 GPU 子进程广播命令。

- **Publisher**（`RemoteRunnerPublisher`）：绑定一个 PUB Socket，将 `Execute`、`Sync`、`GetTorchTensor`、`Destroy` 四种消息广播给所有订阅者。
- **Subscriber**（每个 `RemoteRunnerServer` 内部）：连接一个 SUB Socket，订阅所有消息（`""` 前缀过滤器），非阻塞接收。

**为什么需要 PUB-SUB**：当有 N 个 GPU 子进程时，逐个发送相同的 Execute 命令会产生 N 次序列化开销和 N 次网络往返。PUB-SUB 只需一次序列化和一次 `send_multipart`，所有 Subscriber 同时收到消息。

### 4.2 REQ-REP（请求-响应）

**用途**：Controller 与单个 GPU 子进程进行同步的点对点通信。

- **Requester**（`RemoteRunnerClient`）：绑定一个 REQ Socket，发送请求后阻塞等待响应。
- **Replier**（每个 `RemoteRunnerServer` 内部）：连接一个 REP Socket，接收请求并返回响应。

**为什么需要 REQ-REP**：`GetTorchTensor` 需要从特定子进程获取 Tensor 数据（通过 IPC 共享内存句柄），`Sync` 需要确认子进程的所有 CUDA Stream 已完成。这些操作是点对点的，需要请求-响应模式。

### 4.3 消息顺序保证

PUB-SUB 和 REQ-REP 是两个独立的 Socket，ZMQ **不保证**跨 Socket 的消息顺序。如果直接通过 REQ-REP 发送 `GetTorchTensor`，它可能先于 PUB-SUB 通道中尚未投递的 `Execute` 消息到达，导致在算子执行完成前就尝试获取 Tensor。

DTorch 的解决方案：

1. **所有消息首先通过 PUB-SUB 发送**：`Sync` 和 `GetTorchTensor` 也作为 PUB-SUB 消息广播，确保它们在 Subscriber 端的接收顺序与之前的 `Execute` 消息一致。
2. **REQ-REP 仅用于获取响应**：当 Subscriber 收到 `Sync` 或 `GetTorchTensor` 的 PUB 消息后，`RemoteRunnerServer` 等待对应的 REQ 消息到达，然后通过 REP Socket 返回实际结果。
3. **消息 ID 串联**：每条 PUB 消息带有单调递增的消息 ID，REQ 消息中也嵌入相同的消息 ID。Server 通过比对 ID 确保 REQ 请求与已消费的 PUB 消息匹配。

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

**使用方式**：

| 调用方 | 方法 | 场景 |
|---|---|---|
| `RemoteRunnerPublisher` | `GetIdAndIncrement()` | 发送每条 PUB 消息时，获取新 ID 并递增 |
| `RemoteRunnerClient` | `GetId()` | 发送 REQ 消息时，读取当前 ID（与已 PUB 的消息 ID 一致） |

这样，**每对 PUB 消息和 REQ 消息共享相同的消息 ID**。Server 收到 REQ 时，检查其 ID 是否与最后一条已消费的 PUB 消息 ID 匹配。

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
- Socket 绑定到随机生成的 IPC 地址。

### 6.2 消息类型

| 常量 | 字符串值 | 用途 | 帧数 |
|---|---|---|---|
| `kExecuteStr` | `"publisherExecute"` | 广播 Operator 序列化数据 | 3 |
| `kGetTorchTensorStr` | `"publisherGetTorchTensor"` | 通知即将获取 Tensor（顺序标记） | 2 |
| `kSyncStr` | `"publisherSync"` | 通知即将同步（顺序标记） | 2 |
| `kDestroyStr` | `"publisherDestroy"` | 通知所有 Server 退出 | 2 |

### 6.3 消息帧格式

每条 PUB 消息由 2 或 3 帧组成：

```
Frame 0: messageId    (字符串，如 "0", "1", "2", ...)
Frame 1: messageType  (kExecuteStr / kGetTorchTensorStr / kSyncStr / kDestroyStr)
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

`OperatorSerializationPack` 包含：输入 Operand 的 `uintptr_t`、输出 Operand 的 `uintptr_t`、算子参数（`OpParam`）、唯一 ID。这使得子进程可以完整重建 Operator 对象。

### 6.5 析构时的 Destroy 通知

```cpp
RemoteRunnerPublisher::~RemoteRunnerPublisher() { SendDestroy(); }
```

析构时发送 `kDestroyStr` 消息，通知所有 `RemoteRunnerServer` 退出其 `AsyncMain` 循环。

---

## 7. RemoteRunnerServer — 广播接收端 + 请求响应端

`RemoteRunnerServer` 在每个 GPU 子进程中运行，包含两个 Socket：SUB（订阅 Publisher 广播）和 REP（响应 Client 请求）。它拥有一个 `NaiveRemoteRunner` 引用，负责实际的算子执行。

### 7.1 内部结构

```cpp
struct RemoteRunnerServer::Impl {
    ::zmq::context_t ctx;
    ::zmq::socket_t subscriber;  // SUB socket, connect 到 publisherAddress
    ::zmq::socket_t replyer;     // REP socket, connect 到 serverAddress
    std::string subscriberMessageType;   // 最后一条 SUB 消息的类型
    int64_t subscriberMessageId;         // 最后一条 SUB 消息的 ID
    NaiveRemoteRunner& mNaiveRemoteRunner;
    std::thread asyncThread;
    std::atomic_bool getDestroySignal;
    std::vector<::zmq::message_t> replyerRecvMsgs;  // REP 接收缓冲区
};
```

- `subscriber`：设置 `rcvhwm = 0`（无接收高水位），订阅所有消息（`""` 前缀），设置 `rcvtimeo`。
- `replyer`：设置 `sndtimeo`。
- 构造函数中启动 `AsyncMain` 线程。

### 7.2 AsyncMain 主循环

```
1. ProcessReplyerServerReadyMessage()
   → 阻塞等待 Client 发送的 kServerReadyStr REQ 消息
   → 回复 kServerReadyStr（握手完成）

2. while (!getDestroySignal):
   a. ProcessSubscriberMessage()
      → 非阻塞 (dontwait) 从 SUB Socket 接收消息
      → 校验消息 ID == subscriberMessageId + 1（严格顺序）
      → 根据消息类型分发：
         - kExecuteStr: 反序列化 Operator，调用 NaiveRemoteRunner::ExecuteSerialization()
         - kSyncStr:     调用 NaiveRemoteRunner::Sync()
         - kGetTorchTensorStr: 无操作（等待后续 REQ 消息）
         - kDestroyStr:  设置 getDestroySignal = true
      → 更新 subscriberMessageType 和 subscriberMessageId

   b. ProcessReplyerMessage(messageType, messageId)
      → 非阻塞从 REP Socket 接收消息（如果还没有缓存的 REQ 消息）
      → 校验 REQ 消息中的 ID == publisherMessageId（匹配已消费的 PUB 消息）
      → 根据消息类型分发：
         - kSyncStr:             回复 kSyncReadyStr
         - kGetTorchTensorStr:   获取 Tensor IPC 句柄，回复 [kGetTensorReadyStr, ipcHandle]
```

### 7.3 消息顺序校验

```cpp
// Subscriber 端：校验消息 ID 严格递增
int64_t gotMessageId = std::stoi(GetMsgAsString(buffers[0]));
if (gotMessageId != mImplPtr->subscriberMessageId + 1) {
    DLogFatal() << "receive message with wrong message id";
}

// Replyer 端：校验 REQ 消息 ID 不早于已消费的 PUB 消息
int64_t gotMessageId = std::stoi(GetMsgAsString(replyerRecvMsgs.at(1)));
if (gotMessageId != publisherMessageId) {
    return;  // 不处理，等待对应的 PUB 消息先到达
}
```

这两层校验确保了即使在异步环境下，操作顺序也是严格正确的。

### 7.4 响应常量

| 常量 | 字符串值 | 用途 |
|---|---|---|
| `kServerReadyStr` | `"serverReady"` | 握手确认 |
| `kSyncReadyStr` | `"serverSyncReady"` | Sync 完成确认 |
| `kGetTensorReadyStr` | `"serverGetTensorReady"` | Tensor 数据就绪 |

---

## 8. RemoteRunnerClient — 请求端

`RemoteRunnerClient` 继承自 `RunnerBase`，是 REQ-REP 模式的请求端。它在父进程中运行，每个 GPU 子进程对应一个 Client 实例。

### 8.1 内部结构

```cpp
struct RemoteRunnerClient::Impl {
    std::string publisherAddress;
    ::zmq::context_t ctx;
    ::zmq::socket_t requester;  // REQ socket, bind 模式
};
```

值得注意的是，REQ Socket 使用 **bind**（而非 connect），Server 端的 REP Socket 使用 **connect**。这与传统客户端绑定、服务端连接的直觉相反，但符合 ZMQ 的推荐实践：哪个端的生命周期更稳定，哪个端就 bind。

### 8.2 关键方法

```cpp
// Execute 有意不实现 —— 执行命令通过 PUB-SUB 广播
void Execute(...) override {
    DLogError() << "Execute is called in RemoteRunnerPublisher()";
    DUnsupportedImpl();
}
```

| 方法 | 协议 | 说明 |
|---|---|---|
| `WaitServerReady()` | REQ→REP | 发送 `[kServerReadyStr, messageId]`，等待 `[kServerReadyStr]` 响应 |
| `GetTorchTensor(operand)` | REQ→REP | 序列化 Operand 指针，发送 `[kGetTensorReadyStr, messageId, serializedPtr]`，接收 `[kGetTensorReadyStr, ipcHandle]`，通过 `TorchUtil::FromIpcMemHandle()` 重建 Tensor |
| `Sync()` | REQ→REP | 发送 `[kSyncReadyStr, messageId]`，等待 `[kSyncReadyStr]` 响应 |

所有 REQ 消息中的 `messageId` 来自 `PublishMessageIdManager::GetId()`（不递增），与之前 PUB 消息的 ID 相同。

---

## 9. 完整消息流

以下以一次 `c = a + b` 计算为例，展示跨越 PUB-SUB 和 REQ-REP 两种模式的完整消息流：

```
时间线 →

Parent Process (EagerGraphExecutor)              Child Process (RemoteRunnerServer)
═══════════════════════════════════              ═══════════════════════════════════

1. Execute:
   RemoteRunnerPublisher.Execute(ops)
   → PUB [id=0, "publisherExecute", data] ────→ SUB recv → ProcessSubscriberExecuteMessage()
                                                              → NaiveRemoteRunner.ExecuteSerialization()
                                                              → mNaiveRunner.Execute()

2. GetTorchTensor:
   RemoteRunnerPublisher.GetTorchTensor()
   → PUB [id=1, "publisherGetTorchTensor"] ──→ SUB recv → (无操作，仅标记顺序)

   RemoteRunnerClient.GetTorchTensor(operand)
   → REQ [id=1, "serverGetTensorReady", ptr] ─→ REP recv → ProcessReplyerGetTensorReady()
                                              ←─ REP send [id=1, "serverGetTensorReady", ipcHandle]
   ← REQ recv → TorchUtil::FromIpcMemHandle()

3. Sync:
   RemoteRunnerPublisher.Sync()
   → PUB [id=2, "publisherSync"] ────────────→ SUB recv → mNaiveRemoteRunner.Sync()

   RemoteRunnerClient.Sync()
   → REQ [id=2, "serverSyncReady"] ──────────→ REP recv → ProcessReplyerSyncMessage()
                                              ←─ REP send [id=2, "serverSyncReady"]
   ← REQ recv (sync 完成)

4. Destroy:
   RemoteRunnerPublisher.SendDestroy()
   → PUB [id=N, "publisherDestroy"] ─────────→ SUB recv → getDestroySignal = true
                                                           → AsyncMain 退出
```

**关键时序保证**：PUB 消息 `id=1` 的 `GetTorchTensor` 通知一定在 `id=0` 的 `Execute` 之后到达 SUB Socket（ZMQ PUB-SUB 保证同一 Socket 内的消息顺序）。Server 在收到 `id=1` 的 PUB 消息后，才会接受和响应匹配 `id=1` 的 REQ 消息。这确保了 `GetTorchTensor` 返回的是最新 `Execute` 的结果。

---

## 10. 源码索引

| 组件 | 头文件 | 实现文件 |
|---|---|---|
| ZMQ 工具函数 | `dtorch/external/zmq/zmq.h` | — |
| `RemoteRunnerPublisher` | `dtorch/external/zmq/remote_runner_publisher.h` | `dtorch/external/zmq/remote_runner_publisher.cc` |
| `PublishMessageIdManager` | `dtorch/external/zmq/remote_runner_publisher.h` | — |
| `RemoteRunnerClient` | `dtorch/external/zmq/remote_runner_client.h` | `dtorch/external/zmq/remote_runner_client.cc` |
| `RemoteRunnerServer` | `dtorch/external/zmq/remote_runner_server.h` | `dtorch/external/zmq/remote_runner_server.cc` |
| ZMQ 基础测试 | `dtorch/tests/test_zeromq.cc` | — |
| 端到端集成测试 | `dtorch/tests/test_remote_runner.cc` | — |
