# Stream Race Condition

## 1. PyTorch 中的 CUDA Stream 同步机制

CUDA Stream 是一组在 GPU 上按顺序执行的 CUDA 操作序列。同一个 Stream 内的操作保证按提交顺序串行执行，而不同 Stream 之间的操作是**并行的、无顺序保证的**。PyTorch 对这个设计做了 GPU 算子层面的封装，使得同一个 Stream 内的 PyTorch 算子按序执行，跨 Stream 的算子异步并行。

DTorch 中，`CudaKernelStream` 内部持有一个异步线程（`mAsyncStreamThread`，参见 `kernel_stream.cc:97`）。该异步线程在 `InitInAsyncThread()` 中调用 `at::cuda::setCurrentCUDAStream` 为自己绑定专属的 CUDA Stream，并由 `TorchCudaStreamGuarantee` 确保在整个 Kernel 执行期间 Stream 不被切换（参见 `cuda_kernel_stream.cc:26-31`）。

`dtorch::core::Kernel` 通过 `LaunchKernel()` 被发送到 `CudaKernelStream` 内部的线程安全队列中，由该异步线程逐个取出并在其绑定的 CUDA Stream 上执行。这意味着：

- **同一个 `CudaKernelStream` 内的 Kernel 天然串行**，不存在 data race
- **跨 `CudaKernelStream` 的 Kernel 是并行的**，需要显式同步来保证正确性

---

### 1.1 Stream Event + Stream Wait Event

这是最底层的 CUDA Stream 同步原语，PyTorch 通过 `torch.cuda.Event`（Python）或 `at::cuda::CUDAEvent`（C++）暴露出来。核心流程：

```
1. 在 Stream A 上记录 Event  →  Event 标记了 Stream A 的当前进度点
2. 让 Stream B 等待该 Event  →  Stream B 在 Event 之前阻塞，直到 Stream A 到达该进度点
```

#### Python 示例

```python
s1 = torch.cuda.Stream()  # Stream A
s2 = torch.cuda.Stream()  # Stream B

# Stream A 上执行计算
with torch.cuda.stream(s1):
    A = torch.randn(100, 100, device="cuda")
    B = A @ A

# 在 s1 上记录 event
event = torch.cuda.Event()
event.record(s1)

# Stream B 等待 event（即等待 s1 上 event 之前的所有操作完成）
s2.wait_event(event)
with torch.cuda.stream(s2):
    C = B @ B  # 此时 B 一定已经计算完成
```

#### C++ API（CUDA Driver API）

DTorch 在 `CudaEvent` 中封装了 CUDA Driver API：

```cpp
// dtorch/external/cuda/cuda_event.h
CudaEvent event;
event.Create();                          // cudaEventCreate
event.Record(streamA);                   // cudaEventRecord — 在 streamA 上标记时间点
event.StreamWaitEvent(streamB);          // cudaStreamWaitEvent — streamB 等待该 event
event.Synchronize();                     // cudaEventSynchronize — CPU 端等待 event 完成
```

---

### 1.2 Tensor.record_stream

#### 前提：PyTorch CUDA 缓存分配器（CUDACachingAllocator）与 Stream 的交互

**PyTorch 的 CUDA 显存分配是与 Stream 隔离的**。Tensor 的显存在分配时不属于任何 Stream——`cudaMalloc` 不是 Stream 上的操作，它返回的显存指针对所有 Stream 的 Kernel 都可见。

`CUDACachingAllocator` 会将每个显存块标记为"被某个 Stream 使用中"。当一个 Block 的引用计数归零后，分配器不会立即 `cudaFree`，而是缓存在分配器池中供后续复用。但在多 Stream 编程中，必须确保该 Block 在**所有正在使用它的 Stream 都完成操作之前**，不会被重新分配给其他请求，否则会引发 data race。`record_stream` 正是为此提供保障的核心机制。

`torch.Tensor.record_stream(stream)` 的作用是**告知 CUDACachingAllocator 该 tensor 的显存还被哪个 Stream 使用**，从而防止分配器在 Stream 完成之前回收这块显存。

#### 问题场景：分配器只知道 Creation Stream

CUDACachingAllocator 的一个关键行为是：**它只追踪 tensor 被分配时的 Stream（creation stream）**。分配器保证：一块显存在 creation stream 上所有 pending 操作完成之前，不会被回收。但这个保证仅限于 creation stream——如果 tensor 在**另一个 Stream** 上被使用，分配器对此一无所知。

考虑以下**错误**范式：

```python
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # 创建一个新的 side stream
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)  # A 在 default stream 上创建
with torch.cuda.stream(s):
    B = torch.sum(A)  # A 在 side stream s 上被读取
# 危险！sum() 可能在 normal_() 完成之前就开始执行
```

这里有两个问题：

1. **执行顺序问题**：`torch.sum(A)` 在 side stream `s` 上执行，而 `normal_()` 在 default stream 上执行。不同 Stream 之间没有顺序保证，`sum()` 可能在 `normal_()` 完成之前就开始读 `A`，读到未初始化的数据。
2. **显存回收问题**（即使解决了执行顺序）：`A` 的 creation stream 是 default stream。当 `A` 在 Python 侧引用归零后，分配器只知道 default stream 上的工作，**不知道 `A` 还被 side stream `s` 使用**。分配器可能在 `s` 还在读 `A` 时，就把 `A` 的显存回收并分配给新的 tensor。

#### 正确范式

PyTorch 官方文档给出了正确的多 Stream 使用范式，需要**两步同步**：

```python
cuda = torch.device('cuda')
s = torch.cuda.Stream()

# 1. 在 default stream 上创建并初始化 A
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)

# 2. 让 s 等待 default stream — 确保 normal_() 在 sum() 之前完成
s.wait_stream(torch.cuda.default_stream(cuda))

# 3. 在 s 上使用 A
with torch.cuda.stream(s):
    B = torch.sum(A)

# 4. 告知分配器：A 的显存在 s 完成之前不要回收
A.record_stream(s)
```

两步同步的职责分离：

| 步骤                     | 机制              | 解决的问题                                             |
| ------------------------ | ----------------- | ------------------------------------------------------ |
| `s.wait_stream(default)` | GPU Stream 间同步 | 执行顺序：确保 `normal_()` 在 `sum()` 之前完成         |
| `A.record_stream(s)`     | 告知分配器        | 显存生命周期：防止分配器在 `s` 完成之前回收 `A` 的显存 |

更微妙的情况是：**即使没有读依赖，也需要同步**。因为 `A = torch.empty(...)` 分配到的显存可能是从之前已释放的 tensor 复用的，而那块显存上可能还有 pending 操作未完成：

```python
s = torch.cuda.Stream()
A = torch.empty((100, 100), device=cuda)
s.wait_stream(torch.cuda.default_stream(cuda))  # 仍然需要！确保 A 的显存可以安全写入
with torch.cuda.stream(s):
    A.normal_(0.0, 1.0)  # 在 s 上写入 A
    A.record_stream(s)    # 告知分配器 A 还被 s 使用
```

#### 参考链接

> - [TORCH.TENSOR.RECORD_STREAM](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream)
> - [CUDA SEMANTICS — PyTorch 2.13 documentation](https://docs.pytorch.org/docs/2.13/notes/cuda.html#cuda-streams)

---

### 1.3 tensor.to() 和 tensor.clone() 运行在哪个 Stream 上

理解这两个操作在哪个 Stream 上运行，是避免跨 Stream race condition 的关键。

#### tensor.to()

##### gpu to gpu

gpu 和 gpu 之间拷贝时，在 src gpu stream 上进行。并自动完成 dest gpu stream 和 src gpu stream 的同步。

源码：https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/Copy.cu#L286

##### gpu 和 cpu 之间拷贝

gpu 和 cpu 之间拷贝时，都是在 gpu 所在的 cuda stream 上运行。

源码：https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/Copy.cu#L437

##### 跨线程使用

由于 PyTorch 中已经做好了必要的同步，因此在同一个 Thread 中使用 tensor.to() 不需要额外的同步操作。但是在多线程环境中，由于 `getCurrentCUDAStream` 返回的是 **thread-local** 的变量。这意味着跨线程使用 tensor 时，必须手动确保源 tensor 所在 Stream 的正确性。

```python
# 线程 1
t = torch.randn(100, device="cuda:0")  # 在 Thread1 的 default stream 上创建

# 线程 2
t2 = t.to(device="cuda:1")  # 危险！Thread2 不一定能正确获取 Thread1 的 stream
```

#### tensor.clone()

```python
torch.Tensor.clone()
```

等价于 `torch.empty_like(self).copy_(self)`，即：

1. 在当前 Stream 上分配一个同 shape 的空 tensor（`empty_like`）
2. 在当前 Stream 上执行 `copy_`（`cudaMemcpyAsync` 或等效 kernel）

**`clone()` 在调用者的当前 CUDA Stream 上执行**，不会自动同步其他 Stream。

这意味着如果源 tensor 是在另一个 Stream 上计算的，`clone()` 不会自动等待那个 Stream：

```python
s = torch.cuda.Stream()
A = torch.randn(100, device="cuda")
with torch.cuda.stream(s):
    B = A @ A  # B 在 stream s 上计算

# 回到 default stream
C = B.clone()  # 危险！default stream 不会等待 s 完成
               # C 可能读到 B 的中间计算结果或未初始化数据
```

**正确做法**：

```python
# 方案1: 先同步 stream
torch.cuda.current_stream().wait_stream(s)
C = B.clone()

# 方案2: 直接 sync（CPU 等待，最简单但性能最差）
torch.cuda.synchronize()
C = B.clone()

# 方案3: 在正确的 stream 上 clone
with torch.cuda.stream(s):
    C = B.clone()
```

### 1.4 getCurrentCUDAStream

PyTorch 中经常需要使用到`getCurrentCUDAStream` 函数，其源码如下。需要注意，current_streams 是 thread_local 的变量，因此多线程环境中使用 `getCurrentCUDAStream` 时需要注意拿到的 Stream 是否符合预期。

```c++
// https://github.com/pytorch/pytorch/blob/v2.13.0/c10/cuda/CUDAStream.cpp#L168C1-L168C68
thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// https://github.com/pytorch/pytorch/blob/v2.13.0/c10/cuda/CUDAStream.cpp#L396C1-L401C2
void setCurrentCUDAStream(CUDAStream stream) {
  initCUDAStreamsOnce();
  auto device_index = stream.device_index();
  check_gpu(device_index);
  current_streams[device_index] = stream.id();
}

// https://github.com/pytorch/pytorch/blob/v2.13.0/c10/cuda/CUDAStream.cpp#L386
CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::cuda::SetTargetDevice();
  }
  check_gpu(device_index);
  return CUDAStreamForId(device_index, current_streams[device_index]);
}
```
