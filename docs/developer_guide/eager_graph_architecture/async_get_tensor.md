# 异步获取 Tensor 值

## 概述

DTorch 将"获取 Tensor 值"建模为一个标准的 **Operator**（`GetTensorOp`），通过自定义的 **Promise/Future 机制**支持异步获取 Tensor 值。
用户调用 `tensor.to_torch_async()` 后立即返回 `TensorFuture`，不阻塞 Python 线程，稍后调用 `future.get()` 即可获取 `torch.Tensor`。

---

## 架构设计

### TensorPromise / TensorFuture 机制

DTorch 自定义了一套 Promise/Future 机制用于异步传递 Tensor 值，位于 `dtorch/core/communication/` 目录。其核心由两个抽象基类构成：`TensorPromise`（生产者端，提供 `SetValue` / `GetFuture`）和 `TensorFuture`（消费者端，提供 `Get` / `Wait` / `Valid`）。

根据执行场景自动选择后端实现：

- **Memory 模式**（同进程多线程）：基于 `std::promise` / `std::future`，零额外开销。
- **File 模式**（同机器多进程）：基于 Boost Interprocess 共享内存，通过 `InterprocessMutex` / `InterprocessCondition` 实现跨进程同步。
- **Network 模式**（多机器，预留）。

后端选择由 `GetTensorPromiseTypeFromOperand()` 根据 Operand 的设备类型和 `perDevicePerProcess` 配置决定：CPU 张量始终使用 Memory 模式；GPU 张量在 `perDevicePerProcess=true` 时使用 File 模式。

详细的 API 定义、类体系和实现细节参见 [TensorPromise / TensorFuture — 异步取值机制](../communicate/tensor_communicate.md#4-tensorpromise--tensorfuture--异步取值机制)。

### _GetTensorAsync 实现

`_GetTensorAsync`（`dtorch/api/cpp/functional/tensor_functional.cc`）是异步取值的核心入口。其关键设计是创建一对 **TensorPromise / TensorFuture**：Future 立即返回给调用者供后续取值，Promise 则封装进 `GetTensorParam`，随 `GetTensorOp` 加入计算图，逐级传递到 Worker 线程/进程的 Kernel 执行中，最终在 `Compute()` 里调用 `promise->SetValue(tensor)` 写入结果。

### GetTensorOp

`GetTensorOp` 是一个系统算子（位于 `dtorch/core/operators/system/`），主要特征：

- **输入**: 1 个 Operand（目标 Tensor，必须是 local tensor，非 DTensor）
- **输出**: 0 个（不产生新 Operand）
- **Compute()**: 从输入 Blob 中取出 `torch::Tensor`，调用 `promise->SetValue()` 写入 Promise
- 继承 `SkipDistributedSpecFromPlacementSignature() = true`，跳过分布式推断

### GetTensorParam 的序列化

`GetTensorParam` 的序列化区分不同的模式

- **Memory 模式**: Promise 不参与序列化（同进程内通过 Operator 对象直传）
- **File 模式**: 序列化时将 Promise 类型 + 共享内存文件名写入；反序列化时通过 `CreateTensorPromiseFromSerialized()` 重建 Promise 并 attach 到已有共享内存

### GetTensorOp::Compute

`Compute()` 在 Worker 端被调用，是整个异步取值链路的终点——从输入 Blob 取出 Tensor 值，写入 Promise，从而唤醒在 `future.Get()` 上阻塞的 Client：

```cpp
void GetTensorOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(outputs.size() == 0);

    const auto& param = GetOpParam<GetTensorParam>();
    DAlwaysAssert(param.promise);
    DAlwaysAssert(inputs[0].has_value());

    if (inputs[0].value().is_cuda()) {
        // 确保该设备上所有 CUDA stream 的计算已完成再跨线程/进程传递
        torch::cuda::synchronize(inputs[0].value().get_device());
    }
    param.promise->SetValue(std::make_shared<torch::Tensor>(inputs[0].value()));
}
```

CUDA `synchronize` 是关键步骤：Worker 计算使用的 CUDA stream 可能不同于生产者 stream，跨线程/进程传递张量前必须确保该设备上所有 stream 已完成，否则 Future 端可能拿到未完成计算的数据。

---

## C++ API

### TensorFuture

```cpp
// dtorch/api/cpp/tensor_future.h
class TensorFuture {
public:
    torch::Tensor Get();                // 阻塞获取 Tensor 值
    torch::Tensor Wait(int64_t timeoutMs);  // 带超时等待
    bool Valid() const;                 // 检查是否 ready（非阻塞）
};
```

`Valid()` 语义与 `std::future::valid()` 一致——`Get()` 消费结果后返回 `false`。

### Tensor 接口

```cpp
// dtorch/api/cpp/tensor.h
class Tensor {
    torch::Tensor GetTorchTensor() const;         // 同步获取（内部调用异步路径的 .Get()）
    TensorFuture GetTorchTensorAsync() const;     // 异步获取，立即返回 TensorFuture
};
```

`GetTorchTensor()` 改为调用 `GetTorchTensorAsync().Get()`，行为不变。

### Functional API

```cpp
// dtorch/api/cpp/functional/tensor_functional.h
TensorFuture _GetTensorAsync(const Tensor& input);
```

---

## Python API

```python
# 同步获取（与 PyTorch 一致）
torch_tensor = dtensor.to_torch()

# 异步获取（新增）
future = dtensor.to_torch_async()
# ... 不阻塞，继续执行其他操作 ...
torch_tensor = future.get()       # 阻塞直到结果 ready
# 或
torch_tensor = future.wait(5000)  # 等待最多 5000ms
is_ready = future.is_ready()      # 非阻塞检查是否 ready
```

### await TensorFuture

`TensorFuture` 实现了 `__await__` 协议，支持在 asyncio 协程中直接 `await`。内部通过轮询 `is_ready()` + `asyncio.sleep` 实现非阻塞等待，释放事件循环给其他协程：

```python
import asyncio

async def async_get(tensor):
    future = tensor.to_torch_async()
    result = await future  # 异步等待，不阻塞事件循环
    return result

result = asyncio.run(async_get(dtensor))
```

---

## Graph Sync

Graph Sync（`Graph::Sync()` / `Graph::SyncFuture()`）也采用了相同的 Promise-Future 机制，通过 `VoidPromise` / `VoidFuture` 实现（无需传递张量数据，仅需完成信号）。与 `GetTensorOp` 类似，Sync 被建模为一个系统算符 `SyncOp`（零输入零输出），每个目标设备对应一个 `SyncKernel`。当 kernel 在对应 CUDA stream 上执行时，通过 `event->Synchronize()` + `BoostAsioThreadPool` 异步等待 GPU 完成后，调用 `VoidPromise::SetValue()` 发出信号。`VoidFutureCollect` 聚合所有设备的 `VoidFuture`，在所有设备就绪后解除阻塞。

同样支持 `kMemory`（同进程，`std::promise<void>` / `std::future<void>`）、`kFile`（跨进程，Boost IPC 共享内存）、`kNetwork`（预留）三种后端。API 层提供阻塞版本 `Sync()`（内部调用 `SyncFuture().Get()`）和异步版本 `SyncFuture()`（返回 `VoidFutureCollect`）。

详见 [tensor_communicate.md](../communicate/tensor_communicate.md)。

---

## 相关文件

| 文件                                                          | 说明                                         |
| ------------------------------------------------------------- | -------------------------------------------- |
| `dtorch/core/communication/promise_future/tensor_promise_future.h/cc`        | TensorPromise / TensorFuture 基类 + 工厂函数 |
| `dtorch/core/communication/promise_future/memory_tensor_promise_future.h/cc` | Memory 实现（`std::promise`/`std::future`）  |
| `dtorch/core/communication/promise_future/file_tensor_promise_future.h/cc`   | File/Boost IPC 实现                          |
| `dtorch/core/operators/system/get_tensor_op.h/cc`             | GetTensorParam + GetTensorOp                 |
| `dtorch/api/cpp/tensor_future.h/cc`                           | TensorFuture 公共 API（PIMP）                |
| `dtorch/api/cpp/tensor.h/cc`                                  | Tensor::GetTorchTensorAsync()                |
| `dtorch/api/cpp/functional/tensor_functional.h/cc`            | _GetTensorAsync()                            |
| `dtorch/api/python/py_bind_tensor.h`                          | Python 绑定                                  |
| `python/dtorch/tensor.py`                                     | DTorchTensor.to_torch_async() + TensorFuture |
