# GraphExecutor

DTorch 的 GraphExecutor 负责把用户的 API 调用转换为可执行的 Kernel 并调度执行。本文介绍单机多线程场景下的执行路径：**GraphConstructor → EagerGraphExecutor → PerDeviceThreadNodeRunner → NaiveRunner**。

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             Python API Layer                             │
│                            tensor_a + tensor_b                           │
├──────────────────────────────────────────────────────────────────────────┤
│                             GraphConstructor                             │
│                 Create Operator -> Push Message Queue (async)            │
├──────────────────────────────────────────────────────────────────────────┤
│                            EagerGraphExecutor                            │
│       Message Loop -> LogicalGraph -> Topological Sort -> Dispatch       │
├──────────────────────────────────────────────────────────────────────────┤
│                        PerDeviceThreadNodeRunner                         │
│                  Thin wrapper, delegates to NaiveRunner                  │
├──────────────────────────────────────────────────────────────────────────┤
│                               NaiveRunner                                │
│     Operator -> Kernel -> Blob Management -> KernelStream Scheduling     │
└──────────────────────────────────────────────────────────────────────────┘
```

核心数据流：

1. **GraphConstructor** 接收 Python API 调用，创建 Operator 并异步推入消息队列。
2. **EagerGraphExecutor** 的 AsyncMain 线程消费消息，将 Operator 加入 LogicalGraph，按拓扑序执行。
3. **PerDeviceThreadNodeRunner** 把调用委托给内部的 NaiveRunner，在多机模式下，NodeRunner 才会发挥作用。
4. **NaiveRunner** 将 Operator 转换为 Kernel，通过 KernelStream 发射到设备上执行。

## 2. GraphConstructor — 计算图构造器

`GraphConstructor` 是 Python API 与执行引擎之间的**桥梁**。用户调用算子时：

1. 将算子参数与输入实例化为 Operator
2. 异步发送给 EagerGraphExecutor，**立即返回**输出的 Tensor（不等待计算完成）
3. 维护 Operand 引用计数：Python 侧 Tensor 全部释放后，通知执行器回收该 Operand 的物理内存

GraphConstructor 与 EagerGraphExecutor 之间通过消息队列通信，消息均为**异步**（新增 Operator、释放 Operand、命名等），不阻塞，Python Client 可持续构建计算图。

## 3. EagerGraphExecutor — 图执行器

`EagerGraphExecutor` 是 Single-Controller 架构中的 **Controller** 实现，运行在独立的 AsyncMain 线程中，循环执行：

1. **消费消息** — 取出本批次新增的 Operator 与不再被引用的 Operand
2. **构建 LogicalGraph** — 将新节点加入计算图，按拓扑序生成执行序列
3. **分发执行** — 将算子交给 Runner 执行
4. **清理** — 每个 Operator **只执行一次**，执行完毕立即从图中删除，使显存占用最小化

执行全程异步，仅当用户取值 / 显式同步时才阻塞等待（经 Promise 唤醒调用方）。

Runner 类型由 `perDevicePerProcess` 选项决定：默认关闭时使用线程模式 `PerDeviceThreadNodeRunner`（本文范围）；开启时使用进程模式，见 [集群模式](graph_executor_in_cluster.md)。

## 4. Runner 执行层

Runner 的统一接口只有一个 `Execute()`，负责把 Operator 转为 Kernel 并调度执行：

- **NodeRunnerBase** — 抽象基类，定义统一接口
- **PerDeviceThreadNodeRunner** — 线程模式的薄封装，把调用整体委托给 NaiveRunner
- **NaiveRunner** — 核心执行引擎：为每个 Operator 创建 Kernel、维护 Operand 到 Blob 的映射、调度 KernelStream 执行

线程模式下，多个 Kernel 之间通过**内存**共享中间结果，无需文件或网络传输。

## 5. 完整执行流程

以 `c = a + b` 为例，一次加法经过四层组件的完整路径：

```
Python: c = a + b
    │
GraphConstructor       创建 add 的 Operator，推入消息队列，立即返回 c（异步）
    │
EagerGraphExecutor     AsyncMain 消费消息 → 加入 LogicalGraph → 拓扑排序
    │
Runner                 为每个设备创建 Kernel，准备输入 / 输出 Blob
    │
KernelStream           Kernel 发射到流中，由专用线程异步执行
    │
                       执行完立即释放；Python 取 c 的值时才同步等待
```

## 6. 源文件索引

| 文件 | 说明 |
|---|---|
| `dtorch/core/graph/graph_constructor.h` `.cc` | GraphConstructor — 计算图构造器 |
| `dtorch/core/graph/eager_graph_executor.h` `.cc` | EagerGraphExecutor — 图执行器 |
| `dtorch/core/graph/eager_graph_executor_message.h` | EGEMessage 消息体系 |
| `dtorch/core/graph/logical_graph.h` | LogicalGraph — DAG 容器 |
| `dtorch/core/graph/graph_traversal_sequence.h` | GraphTraversalSequence — 拓扑执行序列 |
| `dtorch/core/runner/node_runner_base.h` | NodeRunnerBase — Runner 基类 |
| `dtorch/core/runner/per_device_thread_node_runner.h` | PerDeviceThreadNodeRunner — 线程 Runner |
| `dtorch/core/runner/naive_runner.h` `.cc` | NaiveRunner — 核心执行引擎 |
| `dtorch/core/runner/runner_supported_devices.h` | RunnerSupportedDevices — 设备支持 |
