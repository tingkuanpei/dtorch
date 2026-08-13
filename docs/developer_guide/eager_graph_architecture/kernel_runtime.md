# Kernel 运行时

LogicalGraph 中的 Operator 只携带元信息。执行时，框架把每个 Operator 转化为若干 **Kernel**（最小执行单元），分配到对应设备的 **KernelStream**（异步执行流）上运行，由 **Blob**（持有 torch::Tensor）承接数据读写。完整链路：

```
构图: 推导元信息 + 确定算子要在哪些设备上执行
        │
        │  为每个 (设备, 流类型) 创建一个 Kernel，准备输入/输出 Blob
        ▼
        Kernel ──► 推入 KernelStream 的队列，立即返回（异步）
        │
        ▼
流线程: 循环取出 Kernel → 执行 → 立即释放（降低峰值显存）
```

## 1. Blob — 张量数据的物理容器

`Blob` 持有 `torch::Tensor`，位于 Operand（元信息）与 Kernel（计算）之间，把逻辑数据节点关联到物理内存。分布式张量在每个设备上各有一个 Blob，按全局设备 ID 索引；Kernel 只访问本设备的 Blob，不跨设备读写。

## 2. Kernel — 最小执行单元

`Kernel` 持有父 Operator、本设备的输入/输出 Blob 以及所属流。`Run()` 依次完成：从输入 Blob 取出 `torch::Tensor` → 调用 `Compute()`（默认委托给 LibTorch 算子）→ 结果写回输出 Blob。部分 Kernel 有特化实现，部分还能在 Kernel 中调用 Python 代码（见 [Python Kernel](python_kernel.md)）。

## 3. KernelStream — 异步执行流

`KernelStream` 封装一个设备上的一条执行流（CPU 线程或 CUDA Stream），由**专用线程**按序执行 Kernel 队列。流分为两类：

| 类型 | 用途 |
|---|---|
| `kCompute` | 常规算子计算（matmul、relu、conv2d 等） |
| `kCommunicate` | 集合通信（all-reduce、all-gather 等） |

同一设备可同时拥有两类流，利用 CUDA Stream 并发实现**计算与通信重叠**。

执行是异步的：Kernel 被推入流的队列后立即返回，流线程循环取 Kernel 执行、执行完立即释放。需要同步时（如取值），等待流中已有 Kernel 全部完成。

## 4. 一个 Operator → 多个 Kernel

一个 Operator 需要在其输入/输出 Operand 涉及的**每个设备**上执行，因此会实例化为多个 Kernel：构图时框架根据 DeviceMesh 确定算子要执行的设备集合；执行时为每个设备创建一个 Kernel，各 Kernel 只负责本设备的计算。

```
Operator（DeviceMesh = {GPU:0, GPU:1}）
   │
   ├──► Kernel(GPU:0) ──► KernelStream(GPU:0)    ← 只访问 GPU 0 的数据
   └──► Kernel(GPU:1) ──► KernelStream(GPU:1)    ← 只访问 GPU 1 的数据
```

## 5. 源文件索引

| 文件 | 说明 |
|---|---|
| `dtorch/core/blob.h` `.cc` | Blob — 张量数据物理容器 |
| `dtorch/core/kernel/kernel.h` `.cc` | Kernel — 最小执行单元 |
| `dtorch/core/kernel_stream/kernel_stream.h` `.cc` | KernelStream — 异步执行流 |
| `dtorch/core/kernel_stream/cpu_kernel_stream.h` / `cuda_kernel_stream.h` `.cc` | CPU / CUDA 流的实现 |
| `dtorch/core/kernel_stream/kernel_stream_manager.h` `.cc` | 流管理器 |
| `dtorch/core/operators/operator_assign_info.h` | 算子到流的分配信息 |
| `dtorch/core/runner/node_runner_base.h` / `naive_runner.cc` | Runner — 创建并分派 Kernel |
