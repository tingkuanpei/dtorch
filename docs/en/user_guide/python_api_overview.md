# Python API Overview

DTorch provides a Python API consistent with PyTorch, with native support for the distributed tensor (DTensor). Users do not need to modify the computation logic of their code — just add the definitions of `DeviceMesh` and `Placements`, and a single-GPU PyTorch program can be seamlessly scaled to a multi-GPU distributed environment.

---

## 1. single-device API matches PyTorch

When `device_mesh` is not specified, DTorch's interfaces are identical to PyTorch's — Tensor construction, factory functions, operators, and the type system all correspond one-to-one, so existing single-GPU PyTorch code runs unchanged:

```python
import dtorch

a = dtorch.randn(4, 8)               # construction / factory functions (zeros, ones, randn, full, arange …)
b = dtorch.matmul(a, a.transpose(0, 1))  # functional operators
c = (a + a * 2.0).relu().sum()       # operator overloading (+ - * / @) + method style + reduce
```

The type system also reuses PyTorch's: `dtorch.float32 is torch.float32`, and `dtype`, `device`, `finfo`/`iinfo`, etc. are all directly referenced from PyTorch.

---

## 2. DTensor

DTorch natively supports the distributed tensor (DTensor), describing a tensor's distribution across devices via `DeviceMesh` and `Placement`. For a DTensor overview, see [Distributed Tensor Overview](distributed_tensor_overview.md).

### Creating a DeviceMesh

```python
from dtorch.distributed_spec import DeviceMesh, init_device_mesh

# Method 1: constructor
#   single device: pass only device_type
mesh = DeviceMesh("cuda")
#   distributed: mesh accepts a nested list (or torch.Tensor) describing the device ids on each dimension
mesh = DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=["dp", "tp"])  # 2D: dp=2, tp=2

# Method 2: via init_device_mesh (recommended)
mesh = init_device_mesh("cuda", mesh_shape=2, mesh_dim_names=["dp"])             # 1D: 2-GPU DP
mesh = init_device_mesh("cuda", mesh_shape=(2, 2), mesh_dim_names=["dp", "tp"])  # 2D: dp=2, tp=2
```

### Creating a DTensor

**All Tensor creation interfaces support the `device_mesh` and `placements` parameters**, including:

- The constructor `dtorch.Tensor(...)`
- The factory function `dtorch.tensor(...)`
- Factory operators such as `dtorch.zeros(...)`, `dtorch.ones(...)`, `dtorch.empty(...)`, `dtorch.rand(...)`, `dtorch.randn(...)`, `dtorch.full(...)`, `dtorch.arange(...)`

The full signature of each interface contains the following optional parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `device_mesh` | `DeviceMesh` | `Graph.default_device_mesh` | The target device mesh. When not specified, the default DeviceMesh of the current Graph is used (usually CPU). |
| `placements` | `Sequence[Placement]` | all `Replicate()` | The list of distribution strategies. When not specified, all dimensions default to `Replicate()`. |

> **Note**: `device` and `device_mesh` are mutually exclusive and cannot be specified at the same time. When the `device` parameter is passed, a single-device `DeviceMesh` is built automatically.

```python
from dtorch import Tensor, DeviceMesh
from dtorch.distributed_spec import (
    init_device_mesh, Shard, Replicate, Partial
)

# Create a DeviceMesh
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])

# Create a DTensor: Shard batch along the dp dim, Replicate along the tp dim
dt = dtorch.zeros(
    4, 128, 256,
    device_mesh=mesh,
    placements=[Shard(0), Replicate()]
)
# shape=(4, 128, 256), device_mesh=cuda(2x2), placements=[S0, R]
```

### Converting to and from a PyTorch Tensor

- From a PyTorch Tensor

```python
torch_t = torch.randn(4, 128)
dt = dtorch.Tensor(torch_t, device_mesh=mesh, placements=[Shard(0), Replicate()])
```

- To a PyTorch Tensor

```python
local_tensor = dt.to_torch()  # returns the torch.Tensor on the current device
```

---

## 3. Operators natively support DTensor

All of DTorch's operators natively support DTensor. When the inputs are DTensors, the framework **automatically infers** the output Tensor's `DeviceMesh` and `Placements` — users do not need to specify them manually, and the code logic stays identical to the single-GPU case.

### DeviceMesh inference

DTorch requires all input/output Tensors of an operator to have the same `DeviceMesh` (i.e., the data must participate in computation on the same group of devices), so the framework directly copies the inputs' `DeviceMesh` to the output.

```python
mesh = init_device_mesh("cuda", 2, mesh_dim_names=["dp"])
a = dtorch.zeros(4, 128, device_mesh=mesh, placements=[Shard(0)])
b = dtorch.zeros(4, 128, device_mesh=mesh, placements=[Replicate()])

c = dtorch.add(a, b)   # input DeviceMeshes match, output DeviceMesh is automatically mesh
```

If the input Tensors' `DeviceMesh`es do not match, align them first with `tensor.redistribute()` (see Section 4).

### Placements inference

Different operators automatically infer the output Placements according to their computation semantics. The following examples show the inference behavior of common operators:

**Element-wise operators (ReLU, SiLU, add, mul...)**:

Placements pass through from the inputs to the output unchanged.

```python
a = dtorch.zeros(4, 128, device_mesh=mesh, placements=[Shard(0)])
b = dtorch.relu(a)      # placements unchanged: [Shard(0)]
c = dtorch.add(a, a)    # placements unchanged: [Shard(0)]
```

**Softmax**:

All dimensions stay unchanged except the softmax dimension, which must be `Replicate()`.

```python
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])
a = dtorch.zeros(4, 128, device_mesh=mesh_2d, placements=[Shard(0), Replicate()])
b = dtorch.nn.functional.softmax(a, dim=-1)
# b.placements = [Shard(0), Replicate()]
```

### When Placements are incompatible

If the combination of input Tensors' Placements is illegal for the current operator, the framework raises an exception whose message contains:

- The operator type
- The actual Placements of each input Tensor
- The list of legal Placements supported by this operator

In this case, the user simply calls `tensor.redistribute()` (see Section 4) to adjust the inputs to a legal distribution before passing them to the operator.

---

## 4. redistribute() on DTensor

Changing a Tensor's Placements in DTorch does **not** require manually creating a `ProcessGroup` like in PyTorch. Just call `tensor.redistribute()` directly — the framework automatically invokes the corresponding collective communication operators (such as AllReduce, AllGather, ReduceScatter, etc.) to redistribute the data across devices.

### tensor.redistribute()

Redistribute the Tensor to the specified DeviceMesh and Placements:

```python
new_tensor = tensor.redistribute(
    device_mesh=new_device_mesh,   # target DeviceMesh
    placements=[Shard(0), Replicate()]  # target Placements
)
```

### tensor.redistribute_like()

Redistribute the Tensor to the same DeviceMesh and Placements as another Tensor:

```python
new_tensor = tensor.redistribute_like(target_tensor)
```

### tensor.redistribute_by_dict()

When the DeviceMesh dimensions are named, Placements can be specified with a dict, which is more intuitive:

```python
# device_mesh dim_names = ["dp", "tp"]
new_tensor = tensor.redistribute_by_dict(
    device_mesh=mesh,                  # optional, defaults to the current DeviceMesh
    placements_dict={
        "dp": Shard(0),               # shard the dp dimension along batch
        "tp": Replicate(),            # replicate the tp dimension
    },
    default_placement_mode="keep",    # for unspecified dimensions: raise_error / replicate / keep
)
```

---

## 5. Async Tensor value retrieval

DTorch supports retrieving Tensor values asynchronously, avoiding the synchronous blocking of `to_torch()`.

### to_torch_async

`tensor.to_torch_async()` immediately returns a `TensorFuture` object without blocking the Python thread:

```python
import dtorch

a = dtorch.rand(1000, 1000)
b = dtorch.rand(1000, 1000)
c = dtorch.matmul(a, b)

# get the result asynchronously
future = c.to_torch_async()
# ... can keep doing other work without being blocked ...

# get the result later
result = future.get()  # blocks until the computation result is ready
```

### ⭐️⭐️⭐️ await TensorFuture

`TensorFuture` implements the `__await__` protocol and can be `await`ed directly inside an asyncio coroutine; internally it polls `is_ready()` with `asyncio.sleep` for non-blocking waiting:

```python
import asyncio
import dtorch

async def async_get(tensor):
    future = tensor.to_torch_async()
    result = await future  # async wait, does not block the event loop
    return result

dtorch_x = dtorch.Tensor(torch.ones(2, 3))
result = asyncio.run(async_get(dtorch_x))
```

### Why PyTorch cannot support await on Tensor

There are two approaches to "retrieving a Tensor's value asynchronously": multi-threading and coroutines.

**Multi-threading**

Restricted by Python's GIL, there is no true multi-threading in Python, and a multi-threading-based async implementation would hit GIL-related performance problems.

**Coroutines**

Coroutines can circumvent the GIL, but PyTorch's API does not natively support coroutines, mainly due to two obstacles:

1. Operators such as `tensor.to` trigger synchronous waits between CPU and GPU, at which point the current coroutine blocks and cannot be released (i.e., cannot finish or be garbage collected), and the event loop cannot switch to other coroutines to continue execution.
2. [The CUDA stream queue has an upper limit on the number of unexecuted kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-scale-launch-queues); once the limit is reached, the thread also blocks waiting until a free slot appears in the queue.

DTorch, on the other hand, supports retrieving Tensor values asynchronously via `await TensorFuture`; thanks to the Client → Controller → Worker asynchronous computation architecture, it does not encounter the two obstacles above.

---

## 6. Easter Egg: Single-Device Distributed Simulation

DTorch supports running distributed programs on a **single GPU** (given sufficient memory) — for example, debugging multi-GPU DP+TP parallel code on a machine with only one GPU.

This feature is off by default; enable it with the environment variable `DTORCH_DTENSOR_IN_SAME_DEVICE=1` (or `=true`); optionally use `DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE` to specify the number of simulated GPUs (default 8). The framework automatically simulates data sharding and collective communication on the single GPU, letting developers **complete the development and debugging of distributed programs on a single GPU**, and deploy to a real multi-GPU cluster after confirming correctness.

This is in sharp contrast to PyTorch: PyTorch's distributed support relies on NCCL for collective communication, and NCCL does not support collective communication between multiple processes on the same GPU, so `init_process_group` requires multiple GPUs to actually exist — a single-GPU environment cannot run multi-GPU distributed code.

---

## 7. Summary

With DTorch's distributed API, users can **extend a single-machine program to a distributed program with minimal code changes** while **keeping the same development and debugging experience**:

- **Computation logic unchanged**: operator calls and the model forward logic are identical to the single-GPU case; only `device_mesh` and `placements` need to be declared when creating Tensors
- **Automatic inference**: the framework automatically infers the distributed information of output Tensors, no manual communication management needed
- **No ProcessGroup needed**: change the distribution strategy declaratively via `redistribute()`, and the framework automatically inserts collective communication operators
