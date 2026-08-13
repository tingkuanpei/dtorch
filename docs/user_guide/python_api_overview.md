# Python API Overview

DTorch 提供与 PyTorch 一致的 Python API，并原生支持分布式张量（DTensor）。用户无需修改代码的计算逻辑，只需增加 DeviceMesh 和 Placements 的定义，即可将单卡 PyTorch 程序无缝扩展到多卡分布式环境。

---

## 1. single-device API 与 PyTorch 一致

当不指定 `device_mesh` 时，DTorch 的接口与 PyTorch 完全一致——Tensor 构造、工厂函数、算子和类型系统都一一对应，已有的单卡 PyTorch 代码无需改动即可运行：

```python
import dtorch

a = dtorch.randn(4, 8)               # 构造 / 工厂函数（zeros、ones、randn、full、arange …）
b = dtorch.matmul(a, a.transpose(0, 1))  # 函数式算子
c = (a + a * 2.0).relu().sum()       # 运算符重载（+ - * / @）+ 方法风格 + reduce
```

类型系统同样复用 PyTorch：`dtorch.float32 is torch.float32`，`dtype`、`device`、`finfo`/`iinfo` 等均直接引用自 PyTorch。

---

## 2. DTensor

DTorch 原生支持分布式张量（DTensor），通过 `DeviceMesh` 和 `Placement` 描述张量在多设备上的分布。DTensor 的概述请参考 [Distributed Tensor Overview](distributed_tensor_overview.md)。

### DeviceMesh 的创建

```python
from dtorch.distributed_spec import DeviceMesh, init_device_mesh

# 方式 1：构造函数创建
#   单设备：只传 device_type
mesh = DeviceMesh("cuda")
#   分布式：mesh 可传嵌套 list（或 torch.Tensor），描述各维度上的设备编号
mesh = DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=["dp", "tp"])  # 2D: dp=2, tp=2

# 方式 2：通过 init_device_mesh 创建（推荐）
mesh = init_device_mesh("cuda", mesh_shape=2, mesh_dim_names=["dp"])             # 1D: 2 卡 DP
mesh = init_device_mesh("cuda", mesh_shape=(2, 2), mesh_dim_names=["dp", "tp"])  # 2D: dp=2, tp=2
```

### DTensor 的创建

**所有 Tensor 创建接口均支持 `device_mesh` 和 `placements` 参数**，包括：

- 构造函数 `dtorch.Tensor(...)`
- 工厂函数 `dtorch.tensor(...)`
- 工厂算子 `dtorch.zeros(...)`、`dtorch.ones(...)`、`dtorch.empty(...)`、`dtorch.rand(...)`、`dtorch.randn(...)`、`dtorch.full(...)`、`dtorch.arange(...)` 等

每个接口的完整签名中包含以下可选参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `device_mesh` | `DeviceMesh` | `Graph.default_device_mesh` | 目标设备网格。不指定时使用当前 Graph 的默认 DeviceMesh（通常为 CPU）。 |
| `placements` | `Sequence[Placement]` | 全 `Replicate()` | 分布策略列表。不指定时所有维度默认为 `Replicate()`。 |

> **注意**：`device` 和 `device_mesh` 参数互斥，不能同时指定。传入 `device` 参数时，会自动构建单设备的 `DeviceMesh`。

```python
from dtorch import Tensor, DeviceMesh
from dtorch.distributed_spec import (
    init_device_mesh, Shard, Replicate, Partial
)

# 创建 DeviceMesh
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])

# 创建 DTensor：沿 dp 维 Shard batch，沿 tp 维 Replicate
dt = dtorch.zeros(
    4, 128, 256,
    device_mesh=mesh,
    placements=[Shard(0), Replicate()]
)
# shape=(4, 128, 256), device_mesh=cuda(2x2), placements=[S0, R]
```

### 与 PyTorch Tensor 的相互转换

- From PyTorch Tensor

```python
torch_t = torch.randn(4, 128)
dt = dtorch.Tensor(torch_t, device_mesh=mesh, placements=[Shard(0), Replicate()])
```

- To PyTorch Tensor

```python
local_tensor = dt.to_torch()  # 返回当前设备上的 torch.Tensor
```

---

## 3. 算子原生支持 DTensor

DTorch 的所有算子均原生支持 DTensor。当输入是 DTensor 时，框架**自动推导**输出 Tensor 的 `DeviceMesh` 和 `Placements`，用户无需手动指定，代码逻辑与单卡完全一致。

### DeviceMesh 推导

DTorch 要求算子的所有输入/输出 Tensor 具有相同的 `DeviceMesh`（即数据必须在同一组设备上参与计算），因此框架直接将输入的 `DeviceMesh` 拷贝给输出。

```python
mesh = init_device_mesh("cuda", 2, mesh_dim_names=["dp"])
a = dtorch.zeros(4, 128, device_mesh=mesh, placements=[Shard(0)])
b = dtorch.zeros(4, 128, device_mesh=mesh, placements=[Replicate()])

c = dtorch.add(a, b)   # 输入 DeviceMesh 一致，输出 DeviceMesh 自动为 mesh
```

如果输入 Tensor 的 `DeviceMesh` 不一致，需先通过 `tensor.redistribute()` 对齐（见第 4 节）。

### Placements 推导

不同算子根据其计算语义自动推导输出 Placements。以下通过示例展示常见算子的推导行为：

**逐元素算子（ReLU、SiLU、add、mul...）**：

Placements 从输入直接透传到输出，不做任何改变。

```python
a = dtorch.zeros(4, 128, device_mesh=mesh, placements=[Shard(0)])
b = dtorch.relu(a)      # placements 保持不变：[Shard(0)]
c = dtorch.add(a, a)    # placements 保持不变：[Shard(0)]
```

**Softmax**：

除 softmax 所在维度必须为 `Replicate()` 外，其余维度不变。

```python
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])
a = dtorch.zeros(4, 128, device_mesh=mesh_2d, placements=[Shard(0), Replicate()])
b = dtorch.nn.functional.softmax(a, dim=-1)
# b.placements = [Shard(0), Replicate()]
```

### Placements 不兼容时

如果输入 Tensor 的 Placements 组合对于当前算子不合法，框架会抛出异常，异常信息包含：

- 算子类型
- 各输入 Tensor 的实际 Placements
- 该算子支持的合法 Placements 列表

此时用户只需调用 `tensor.redistribute()`（见第 4 节）将输入调整到合法分布后再传入算子即可。

---

## 4. DTensor 的 redistribute()

DTorch 中改变 Tensor 的 Placements **不需要**像 PyTorch 那样手动创建 `ProcessGroup`。直接调用 `tensor.redistribute()` 即可，框架会自动调用对应的集合通讯算子（如 AllReduce、AllGather、ReduceScatter 等）完成数据在设备间的重分布。

### tensor.redistribute()

将 Tensor 重分布到指定的 DeviceMesh 和 Placements：

```python
new_tensor = tensor.redistribute(
    device_mesh=new_device_mesh,   # 目标 DeviceMesh
    placements=[Shard(0), Replicate()]  # 目标 Placements
)
```

### tensor.redistribute_like()

将 Tensor 重分布为与另一个 Tensor 相同的 DeviceMesh 和 Placements：

```python
new_tensor = tensor.redistribute_like(target_tensor)
```

### tensor.redistribute_by_dict()

当 DeviceMesh 的维度有命名时，可以使用字典方式指定 Placements，更加直观：

```python
# device_mesh 的 dim_names = ["dp", "tp"]
new_tensor = tensor.redistribute_by_dict(
    device_mesh=mesh,                  # 可选，默认使用当前 DeviceMesh
    placements_dict={
        "dp": Shard(0),               # dp 维度沿 batch 切分
        "tp": Replicate(),            # tp 维度复制
    },
    default_placement_mode="keep",    # 未指定的维度：raise_error / replicate / keep
)
```

---

## 5. 异步获取 Tensor 值

DTorch 支持异步获取 Tensor 值，避免 `to_torch()` 的同步阻塞。

### to_torch_async

`tensor.to_torch_async()` 立即返回一个 `TensorFuture` 对象，不阻塞 Python 线程：

```python
import dtorch

a = dtorch.rand(1000, 1000)
b = dtorch.rand(1000, 1000)
c = dtorch.matmul(a, b)

# 异步获取结果
future = c.to_torch_async()
# ... 可以继续执行其他操作，不会被阻塞 ...

# 稍后获取结果
result = future.get()  # 阻塞直到计算结果 ready
```

### ⭐️⭐️⭐️ await TensorFuture

`TensorFuture` 实现了 `__await__` 协议，支持在 asyncio 协程中直接 `await`，内部通过轮询 `is_ready()` + `asyncio.sleep` 实现非阻塞等待：

```python
import asyncio
import dtorch

async def async_get(tensor):
    future = tensor.to_torch_async()
    result = await future  # 异步等待，不阻塞事件循环
    return result

dtorch_x = dtorch.Tensor(torch.ones(2, 3))
result = asyncio.run(async_get(dtorch_x))
```

### PyTorch 中无法支持 await Tensor

支持「异步获取 Tensor 的值」有两种方案：多线程与协程。

**多线程**

受 Python GIL 限制，Python 中不存在真正的多线程，基于多线程实现异步会遇到 GIL 导致的性能问题。

**协程**

协程可以规避 GIL，但 PyTorch 的 API 并未原生支持协程，主要有两个障碍：

1. 诸如 `tensor.to` 等算子会触发 CPU 与 GPU 之间的同步等待，此时当前协程因阻塞而无法释放（即无法结束或被回收），事件循环也就无法切换到其他协程继续执行。
2. [CUDA 流队列中可容纳的未执行 kernel 数量存在上限](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-scale-launch-queues)，一旦达到该上限，线程同样会阻塞等待，直到队列中出现空闲槽位。

而 DTorch 通过 `await TensorFuture` 支持异步获取 Tensor 的值；同时得益于 Client → Controller → Worker 的异步计算架构，不会遇到上述两个障碍。

---

## 6. 彩蛋：单卡模拟分布式

DTorch 支持在**单张 GPU** 上运行分布式程序（显存足够的前提下），例如在只有一张 GPU 的机器上调试多卡 DP+TP 并行的代码。

该功能默认关闭，需通过环境变量 `DTORCH_DTENSOR_IN_SAME_DEVICE=1`（或 `=true`）开启；可选地用 `DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE` 指定模拟的 GPU 数量（默认 8）。框架在单卡上自动模拟数据切分与集合通信，让开发者**在单卡上完成分布式程序的开发与调试**，确认无误后再部署到真实多卡集群。

这与 PyTorch 形成鲜明对比：PyTorch 的分布式依赖 NCCL 进行集合通信，而 NCCL 不支持同一张 GPU 上多进程之间的集合通讯，因此 `init_process_group` 要求实际存在多张 GPU，单卡环境无法运行多卡分布式代码。

---

## 7. 总结

基于 DTorch 的分布式 API，用户可以**以最小的代码改动**将单机程序拓展为分布式程序，并**保持相同的开发与调试体验**：

- **代码逻辑不变**：算子调用、模型 forward 逻辑与单卡完全一致，仅需在 Tensor 创建时声明 `device_mesh` 和 `placements`
- **自动推导**：框架自动推导输出 Tensor 的分布式信息，无需手动管理通信
- **无需 ProcessGroup**：通过 `redistribute()` 声明式地改变分布策略，框架自动插入集合通讯算子
