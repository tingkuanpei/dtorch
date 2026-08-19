# GraphExecutor

DTorch's GraphExecutor converts the user's API calls into executable Kernels and schedules their execution. This article covers the execution path in the single-machine multi-thread scenario: **GraphConstructor → EagerGraphExecutor → PerDeviceThreadNodeRunner → NaiveRunner**.

## 1. Architecture Overview

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

The core data flow:

1. **GraphConstructor** receives Python API calls, creates Operators and pushes them into the message queue asynchronously.
2. The AsyncMain thread of **EagerGraphExecutor** consumes messages, adds Operators to the LogicalGraph, and executes in topological order.
3. **PerDeviceThreadNodeRunner** delegates the calls to its internal NaiveRunner; the NodeRunner only really comes into play in multi-machine mode.
4. **NaiveRunner** converts Operators into Kernels and launches them on the devices through KernelStreams.

## 2. GraphConstructor — the computation graph constructor

`GraphConstructor` is the **bridge** between the Python API and the execution engine. When the user calls an operator:

1. The operator's parameters and inputs are instantiated into an Operator
2. It is sent to the EagerGraphExecutor asynchronously, and the output Tensor is **returned immediately** (without waiting for the computation to complete)
3. Operand reference counting is maintained: once the Python-side Tensors are all released, the executor is notified to reclaim the Operand's physical memory

GraphConstructor and EagerGraphExecutor communicate through a message queue; all messages are **asynchronous** (add Operator, release Operand, naming, etc.), never blocking, so the Python Client can keep building the computation graph.

## 3. EagerGraphExecutor — the graph executor

`EagerGraphExecutor` is the **Controller** implementation in the Single-Controller architecture, running on an independent AsyncMain thread that loops:

1. **Consume messages** — take the Operators newly added in this batch and the Operands that are no longer referenced
2. **Build the LogicalGraph** — add the new nodes to the computation graph, generate the execution sequence in topological order
3. **Dispatch execution** — hand the operators to the Runner for execution
4. **Clean up** — each Operator **executes only once** and is removed from the graph immediately after execution, minimizing memory usage

Execution is asynchronous throughout; it blocks and waits only when the user retrieves a value / explicitly synchronizes (waking the caller through a Promise).

The Runner type is decided by the `perDevicePerProcess` option: when disabled (default) the thread-mode `PerDeviceThreadNodeRunner` is used (the scope of this article); when enabled, process mode is used, see [cluster mode](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/graph_executor_in_cluster/).

## 4. The Runner execution layer

Runners share one unified interface, `Execute()`, responsible for converting Operators into Kernels and scheduling their execution:

- **NodeRunnerBase** — the abstract base class, defining the unified interface
- **PerDeviceThreadNodeRunner** — a thin wrapper of thread mode, delegating the calls entirely to NaiveRunner
- **NaiveRunner** — the core execution engine: creates Kernels for each Operator, maintains the Operand-to-Blob mapping, and schedules KernelStream execution

In thread mode, intermediate results are shared between Kernels through **memory**, with no file or network transfer needed.

## 5. The complete execution flow

Taking `c = a + b` as an example, the complete path of one addition through the four layers of components:

```
Python: c = a + b
    │
GraphConstructor       creates the add Operator, pushes it into the message queue, returns c immediately (async)
    │
EagerGraphExecutor    AsyncMain consumes the message → adds it to the LogicalGraph → topological sort
    │
Runner                creates Kernels for each device, prepares the input / output Blobs
    │
KernelStream          Kernels are launched into the stream, executed asynchronously by dedicated threads
    │
                      released immediately after execution; Python waits synchronously only when retrieving c's value
```

## 6. Source file index

| File | Description |
|---|---|
| `dtorch/core/graph/graph_constructor.h` `.cc` | GraphConstructor — the computation graph constructor |
| `dtorch/core/graph/eager_graph_executor.h` `.cc` | EagerGraphExecutor — the graph executor |
| `dtorch/core/graph/eager_graph_executor_message.h` | the EGEMessage message system |
| `dtorch/core/graph/logical_graph.h` | LogicalGraph — the DAG container |
| `dtorch/core/graph/graph_traversal_sequence.h` | GraphTraversalSequence — the topology execution sequence |
| `dtorch/core/runner/node_runner_base.h` | NodeRunnerBase — the Runner base class |
| `dtorch/core/runner/per_device_thread_node_runner.h` | PerDeviceThreadNodeRunner — the thread Runner |
| `dtorch/core/runner/naive_runner.h` `.cc` | NaiveRunner — the core execution engine |
| `dtorch/core/runner/runner_supported_devices.h` | RunnerSupportedDevices — device support |
