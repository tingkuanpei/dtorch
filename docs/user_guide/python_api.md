# DTorch Python API

DTorch 提供与 PyTorch 一致的 Python API，并原生支持分布式张量（DTensor）。用户无需修改代码的计算逻辑，只需增加 DeviceMesh 和 Placements 的定义，即可将单卡 PyTorch 程序无缝扩展到多卡分布式环境。

---

## 1. 单机单卡 API 与 PyTorch 一致

DTorch 在单机单卡场景下提供与 PyTorch 完全一致的接口体验。

### Tensor 创建

DTorch 的 Tensor 创建方式与 PyTorch 相同，支持直接传入 Python 数据、`torch.Tensor` 或使用工厂函数：

```python
import dtorch

# 从 Python 数据创建（与 PyTorch 一致）
a = dtorch.Tensor([1.0, 2.0, 3.0])
b = dtorch.tensor([[1, 2], [3, 4]], dtype=dtorch.float32)

# 从 torch.Tensor 创建
import torch
c = dtorch.Tensor(torch.randn(3, 4))

# 工厂函数（与 PyTorch 一致）
d = dtorch.zeros(2, 3)
e = dtorch.ones(3, 4)
f = dtorch.empty(2, 2)
g = dtorch.rand(3, 3)
h = dtorch.randn(2, 4)
i = dtorch.full((2, 3), 3.14)
j = dtorch.arange(0, 10)
k = dtorch.from_numpy(np.array([1, 2, 3]))
```

Tensor 构造函数的签名与 PyTorch 兼容：`dtorch.Tensor(*args, device=None, dtype=None, ...)`。传入 `device` 参数可将 Tensor 放置到指定设备（CPU 或 CUDA）：

```python
a = dtorch.Tensor([1.0, 2.0], device="cuda")
b = dtorch.zeros(2, 3, device="cuda:0")
```

**DTorch Tensor 相比 PyTorch Tensor 的增强能力：**

- `tensor.device_mesh` — 获取分布式设备网格（见第 2 节）
- `tensor.placements` — 获取分布策略列表（见第 2 节）
- `tensor.to_torch()` — 将 DTorch Tensor 转换为 `torch.Tensor`（同步阻塞）
- `tensor.to_torch_async()` — 异步获取 Tensor 值，立即返回 `TensorFuture`（见第 9 节）
- `tensor.redistribute(...)` — 改变分布策略（见第 4 节）

### 算子调用

DTorch 的所有算子调用方式与 PyTorch 一致，支持函数式 API 和 Tensor 方法两种风格：

**函数式 API：**

```python
import dtorch

# 二元运算
c = dtorch.add(a, b)
c = dtorch.sub(a, b)
c = dtorch.mul(a, b)
c = dtorch.matmul(a, b)

# 激活函数
c = dtorch.nn.functional.relu(a)
c = dtorch.nn.functional.silu(a)
c = dtorch.nn.functional.gelu(a)
c = dtorch.nn.functional.softmax(a, dim=-1)

# 形状操作
c = dtorch.reshape(a, (2, 3))
c = dtorch.transpose(a, 0, 1)
c = dtorch.cat([a, b], dim=0)
c = dtorch.squeeze(a)

# 数学运算
c = dtorch.exp(a)
c = dtorch.log(a)
c = dtorch.rsqrt(a)

# 规约操作
c = dtorch.sum(a, dim=0)
c = dtorch.mean(a, dim=-1)
c = dtorch.max(a, dim=0)
```

**Tensor 方法风格：**

```python
c = a.matmul(b)
c = a.add(b)
c = a.mul(2.0)
c = a.reshape(2, 3)
c = a.transpose(0, 1)
c = a.sum(dim=0)
```

**算术运算符重载：**

```python
c = a + b      # 等价于 dtorch.add(a, b)
c = a - b      # 等价于 dtorch.sub(a, b)
c = a * b      # 等价于 dtorch.mul(a, b)
c = a / b      # 等价于 dtorch.div(a, b)
c = -a         # 等价于 dtorch.neg(a)
c = a @ b      # 等价于 dtorch.matmul(a, b)
```

### 类型系统

DTorch 复用了 PyTorch 的类型系统，`dtype`、`device`、`finfo`/`iinfo` 等均直接引用自 PyTorch：

```python
dtorch.float32 is torch.float32   # True
dtorch.device is torch.device     # True
dtorch.finfo is torch.finfo       # True
```

并提供了 PyTorch 风格的 Tensor 类型别名：

```python
dtorch.FloatTensor    # dtype 为 float32 的 Tensor 子类
dtorch.DoubleTensor   # dtype 为 float64
dtorch.IntTensor      # dtype 为 int32
dtorch.LongTensor     # dtype 为 int64
```

---

## 2. 原生 DTensor 支持

DTorch 原生支持分布式张量（DTensor），通过 `DeviceMesh` 和 `Placement` 描述张量在多设备上的分布。DTensor 的完整设计理念与机制详解请参考 [distributed_tensor.md](distributed_tensor.md)。

### DeviceMesh — 设备网格

`DeviceMesh` 描述集群 GPU 拓扑，是一个 N 维设备网格。每个维度可以有名称，用于后续 Placement 推导和重分布。

**创建 DeviceMesh：**

```python
from dtorch.distributed_spec import DeviceMesh, init_device_mesh

# 方式 1：单设备（本地模式）
mesh = DeviceMesh("cuda")

# 方式 2：通过 init_device_mesh 创建（推荐）
# 参数：device_type, mesh_shape, mesh_dim_names
mesh = init_device_mesh("cuda", 2, mesh_dim_names=["dp"])          # 1D: 2 卡 DP
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])  # 2D: dp=2, tp=2
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=["dp", "cp"])  # 2D: dp=2, cp=4
```

**DeviceMesh 主要属性：**

| 属性/方法 | 说明 |
|---|---|
| `mesh.device_type` | 设备类型 (`torch.device`) |
| `mesh.shape` | 网格形状，如 `(2, 2)` |
| `mesh.ndim` | 网格维度数 |
| `mesh.dim_names` | 维度名称列表，如 `["dp", "tp"]` |
| `mesh.is_distributed` | 是否分布式 |
| `mesh.dim_name_index(name)` | 根据名称查找维度索引 |
| `mesh.first_device()` | 返回第一个设备 |
| `mesh.unbind(dims)` | 按维度拆分 DeviceMesh |

### Placement — 分布策略

`Placement` 描述 Tensor 在某一个 DeviceMesh 维度上的分布方式。DTorch 提供三种 Placement：

| Placement | 含义 | 示例 |
|---|---|---|
| `Shard(dim)` | 沿 Tensor 的第 `dim` 轴切分到不同设备 | `Shard(0)` — 沿 batch 维度切分 |
| `Replicate()` | 在所有设备上完整复制 | 不切分 |
| `Partial()` | 各设备持有部分结果（如矩阵乘法的部分和） | 需要 reduce 后才能获得完整值 |

### 创建 DTensor

**所有 Tensor 创建接口均支持 `device_mesh` 和 `placements` 参数**，包括：

- 构造函数 `dtorch.Tensor(...)`
- 工厂函数 `dtorch.tensor(...)`
- 工厂算子 `dtorch.zeros(...)`、`dtorch.ones(...)`、`dtorch.empty(...)`、`dtorch.rand(...)`、`dtorch.randn(...)`、`dtorch.full(...)`、`dtorch.arange(...)` 等

每个接口的完整签名中包含以下可选参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `device_mesh` | `DeviceMesh` | `Graph.default_device_mesh` | 目标设备网格。不指定时使用当前 Graph 的默认 DeviceMesh（通常为单设备 CPU）。 |
| `placements` | `Sequence[Placement]` | 全 `Replicate()` | 分布策略列表。不指定时所有维度默认为 `Replicate()`（即所有设备上完整复制）。 |

> **注意**：`device` 和 `device_mesh` 互斥，不能同时指定。传入 `device` 参数时，会自动构建单设备的 `DeviceMesh`。

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

**`placements` 列表长度必须等于 `device_mesh.ndim`**，第 `i` 个 Placement 对应 DeviceMesh 的第 `i` 维。

**不传参数时的默认行为：**

```python
# 不传 device_mesh 和 placements — 使用默认值，等价于普通 PyTorch Tensor
a = dtorch.zeros(4, 128)        # device_mesh: 单设备 CPU, placements: [Replicate()]
b = dtorch.Tensor([1.0, 2.0])   # 同上

# 只传 device — 自动构建单设备 DeviceMesh，placements 默认 Replicate
c = dtorch.zeros(4, 128, device="cuda")  # device_mesh: cuda:0, placements: [Replicate()]
```

**从已有 Tensor 转换为 DTensor：**

```python
# 从 torch.Tensor 创建带分布信息的 DTorch Tensor
torch_t = torch.randn(4, 128)
dt = dtorch.Tensor(torch_t, device_mesh=mesh, placements=[Shard(0), Replicate()])
```

### 打印与取值

DTorch Tensor 的 `__repr__` 会自动展示分布式信息：

```python
>>> print(dt)
dtorch tensor([[0.1234, ..., 0.5678],
               ...,
               [0.9012, ..., 0.3456]])
shape=(4, 128, 256), dtype=torch.float32,
device_mesh=DeviceMesh('cuda', dim_name: ['dp', 'tp'], shape: (2, 2), data: [[0,1],[2,3]]),
placements=[S0, R]
```

**获取本地 `torch.Tensor` 值：**

```python
local_tensor = dt.to_torch()  # 返回当前设备上的 torch.Tensor
```

**访问分布式信息：**

```python
dt.device_mesh    # DeviceMesh 对象
dt.placements     # [Shard(0), Replicate()]
dt.shape          # 全局逻辑 shape
dt.dtype          # 数据类型
dt.is_dtensor     # 是否为分布式 Tensor（True/False）
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

除 softmax 所在维度必须为 `Replicate()` 外，其余维度 Shard 透传。

```python
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])
a = dtorch.zeros(4, 128, device_mesh=mesh_2d, placements=[Shard(0), Shard(1)])
b = dtorch.nn.functional.softmax(a, dim=-1)
# b.placements = [Shard(0), Replicate()]  ← dim=-1 是 softmax 维度，Shard(1) 被强制转为 Replicate()
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
    default_placement_mode="raise_error",  # 未指定的维度：raise_error / replicate / keep
    convert_shard_size_one_to_replicate=True,  # shape 为 1 的 Shard 自动转为 Replicate
)
```

`placements_dict` 的 key 可以是维度名称（字符串）或维度索引（整数）。`default_placement_mode` 控制未在 dict 中指定的维度：

| 模式 | 行为 |
|---|---|
| `"raise_error"` (默认) | 抛出 KeyError |
| `"replicate"` | 未指定的维度默认 Replicate |
| `"keep"` | 保持该维度原有的 Placement |

### Module 的 redistribute_input / redistribute_output

`Module` 基类提供了 `redistribute_input()` 和 `redistribute_output()` 钩子，在 `forward` 执行前后自动调用。子类可以重写这两个方法来实现透明的输入/输出重分布。

**基类接口（`python/dtorch/nn/modules/module.py`）：**

```python
class Module:
    def redistribute_input(self, *args, **kwargs):
        """可被子类重写，返回 (args, kwargs) 元组"""
        return args, kwargs

    def redistribute_output(self, output):
        """可被子类重写，返回重分布后的 output"""
        return output

    def __call__(self, *args, **kwargs):
        # 1. 调用 redistribute_input 重分布输入
        args, kwargs = self.redistribute_input(*args, **kwargs)
        # 2. 执行 forward
        output = self.forward(*args, **kwargs)
        # 3. 调用 redistribute_output 重分布输出
        output = self.redistribute_output(output)
        return output
```

**使用示例（LlamaForCausalLM）：**

```python
class LlamaForCausalLM(Module):
    def redistribute_input(self, input_ids):
        # 保存原始 placement 以便输出时恢复
        self.input_device_mesh = input_ids.device_mesh
        self.input_placement = input_ids.placements

        # 将输入重分布为模型的期望分布：dp → Shard(0), tp → Replicate
        input_ids = input_ids.redistribute_by_dict(
            self.first_param_device_mesh(),
            placements_dict={"dp": Shard(0), "tp": Replicate()},
        )
        return [input_ids], {}

    def redistribute_output(self, logits):
        # 恢复为输入的原始分布
        return logits.redistribute(
            self.input_device_mesh, placements=self.input_placement
        )
```

---

## 5. Linear 算子如何支持 DP、TP、CP

DTorch 的 `Linear` 模块（`python/dtorch/nn/modules/linear.py`）原生支持 DP、TP、CP 等多种并行策略。其核心原则是：**只有 TP 维度需要切分 Weight，DP 和 CP 维度上 Weight 始终保持完整复制（`Replicate()`）**。

### 核心参数：tp_dim 与 tp_shard_type

`Linear` 的构造函数签名：

```python
Linear(in_features, out_features, bias=True, device=None, dtype=None,
       device_mesh=None, *, tp_dim="tp", tp_shard_type=None)
```

**`tp_dim`** — 指定在 DeviceMesh 的哪个维度上执行张量并行（TP）权重切分：

| 取值类型 | 含义 | 示例 |
|---|---|---|
| `str`（默认 `"tp"`） | 匹配 `device_mesh.dim_names` 中同名的维度 | `tp_dim="tp"` → 在名为 `"tp"` 的维度上切分 |
| `int` | 直接指定 DeviceMesh 的维度索引 | `tp_dim=1` → 在第 1 维上切分 |
| `None` | 不做 TP 切分，所有权重保持 Replicate | `ReplicateParallelLinear` 即设 `tp_dim=None` |

> **关键行为**：当 `tp_dim` 是字符串时，调用 `device_mesh.dim_name_index(tp_dim)` 查找匹配的维度。**如果 DeviceMesh 中不存在该名称的维度**（例如 DeviceMesh 只有 `"dp"` 和 `"cp"` 而没有 `"tp"`），返回 `None`，**不会执行 TP 切分**。

**`tp_shard_type`** — 指定权重的切分方向：

| tp_shard_type | 权重 Placement（在 tp_dim 上） | bias Placement（在 tp_dim 上） | 含义 |
|---|---|---|---|
| `"col"` | `Shard(0)` | `Shard(0)` | 按 output features 切分，每个设备持有部分输出列 |
| `"row"` | `Shard(1)` | `Partial()` | 按 input features 切分，每个设备持有部分输入行 |

### 权重切分规则

`Linear` 初始化时，所有权重和 bias 的初始 Placement 在**所有维度**上均为 `Replicate()`。仅 `tp_dim` 对应的维度被替换为切分 Placement：

```python
weight_placements = [Replicate()] * device_mesh.ndim   # 所有维度初始为 Replicate
bias_placements   = [Replicate()] * device_mesh.ndim

if tp_dim is not None:
    weight_placements[tp_dim] = Shard(1) if tp_shard_type == "row" else Shard(0)
    bias_placements[tp_dim]   = Partial()  if tp_shard_type == "row" else Shard(0)
```

**这天然保证了 DP、CP 等维度的兼容性**：因为只有 `tp_dim` 匹配的维度会切分，其余维度（如 `"dp"`、`"cp"`）始终保持 `Replicate()`，不会受到 TP 逻辑的影响。

```
假设 DeviceMesh dim_names = ["dp", "tp", "cp"]，tp_dim="tp"

          dp 维          tp 维           cp 维
weight: [Replicate(), Shard(0|1),  Replicate()]
          ↑ 数据并行     ↑ TP 切分      ↑ 上下文并行
          完整复制       唯一被修改      完整复制
```

### redistribute_input / redistribute_output — 输入输出校验与转换

`Linear` 在 `forward` 前后分别调用 `redistribute_input` 和 `redistribute_output`，对输入和输出 Tensor 进行校验与可选的 Placements 转换。两者都只操作 `tp_dim` 对应的维度，其余维度通过 `default_placement_mode="keep"` 保持不变。

**redistribute_input** — 执行两个任务：

1. **可选转换**：如果调用方传入了 `input_placement`，将输入在 `tp_dim` 上重分布到目标 Placement。
2. **校验**：断言输入在 `tp_dim` 上的 Placement 与权重切分方式匹配。

```python
def redistribute_input(self, input, input_placement=None):
    if self.tp_dim is not None and input_placement is not None:
        input = input.redistribute_by_dict(
            placements_dict={self.tp_dim: input_placement},
            default_placement_mode="keep",
        )
    if self.tp_dim is not None:
        expect = Shard(input.dim() - 1) if self.tp_shard_type == "row" else Replicate()
        assert input.check_placement(self.tp_dim, expect)
    return [input], {}
```

| tp_shard_type | 要求的输入 Placement（tp_dim 上） | 原因 |
|---|---|---|
| `"col"` | `Replicate()` | 每个设备需要完整的输入才能计算各自的部分输出 |
| `"row"` | `Shard(input.ndim - 1)` | 输入在 hidden 维度切分，与权重的 in_features 切分对齐 |

**redistribute_output** — 将输出在 `tp_dim` 上转换为指定的 Placement。基类默认不做转换（`output_placement=None` 时直接返回），子类通过传入特定值实现自动转换：

```python
def redistribute_output(self, output, output_placement=None):
    if self.tp_dim is not None and output_placement is not None:
        output = output.redistribute_by_dict(
            placements_dict={self.tp_dim: output_placement},
            default_placement_mode="keep",
        )
    return output
```

例如 `RowParallelLinearWithReplicateOutput` 调用 `redistribute_output(output, Replicate())`，在 RowParallel 产生 `Partial()` 输出后自动插入 AllReduce 转为 `Replicate()`。

### 便捷子类

基于 `tp_dim` 和 `tp_shard_type` 的组合，DTorch 提供了以下预置子类，覆盖常见并行场景：

| 类 | tp_dim | tp_shard_type | redistribute_input | redistribute_output |
|---|---|---|---|---|
| `ColumnParallelLinear` | `"tp"` | `"col"` | 校验输入为 Replicate | 不做转换 |
| `ColumnParallelLinearWithReplicateOutput` | `"tp"` | `"col"` | 校验输入为 Replicate | 输出转为 Replicate |
| `ColumnParallelLinearWithReplicateInputOutput` | `"tp"` | `"col"` | 输入转为 Replicate | 输出转为 Replicate |
| `RowParallelLinear` | `"tp"` | `"row"` | 校验输入为 Shard(-1) | 不做转换 |
| `RowParallelLinearWithReplicateOutput` | `"tp"` | `"row"` | 校验输入为 Shard(-1) | 输出转为 Replicate |
| `ReplicateParallelLinear` | `None` | — | 不做校验 | 不做转换 |

其中最常用的是 `ColumnParallelLinear` + `RowParallelLinearWithReplicateOutput` 组合（见第 6 节 Llama 示例）。

---

## 6. Llama 模型示例：DP + TP 支持

以下以 Llama 模型为例，展示如何使用 DTorch 的 `DeviceMesh` 和 `Module` 体系实现 DP 和 TP。

### 6.1 模型定义

完整代码见 `python/dtorch/test/modules/llama.py`。

**LlamaForCausalLM** — 顶层模型，指定 DeviceMesh 并配置分布式策略：

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config, device_mesh=None):
        super().__init__()
        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        device_mesh.check_all_dim_names_in_set({"dp", "tp"})

        with Graph.default_graph().device_mesh_guard(device_mesh):
            self.model = LlamaModel(config)
            self.lm_head = nn.ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False,
            )

    def redistribute_input(self, input_ids):
        # 保存输入原始的 placements，输出时恢复
        self.input_device_mesh = input_ids.device_mesh
        self.input_placement = input_ids.placements

        # 将输入重分布到模型期望的分布
        input_ids = input_ids.redistribute_by_dict(
            self.first_param_device_mesh(),
            placements_dict={"dp": Shard(0), "tp": Replicate()},
        )
        return [input_ids], {}

    def redistribute_output(self, logits):
        # 恢复为输入的原始分布
        return logits.redistribute(
            self.input_device_mesh, placements=self.input_placement
        )
```

关键点：
- 使用 `device_mesh_guard` 上下文管理器，将 DeviceMesh 传递给所有子模块
- `check_all_dim_names_in_set({"dp", "tp"})` 确保 DeviceMesh 的维度名不超出支持范围
- `redistribute_input` 将输入从任意分布自动转换为模型期望的分布
- `redistribute_output` 将输出恢复为输入时的分布，保证对外透明

**LlamaSdpaAttention** — 使用 ColumnParallel + RowParallel 实现 TP：

```python
class LlamaSdpaAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        ...
        self.q_proj = nn.ColumnParallelLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias,
        )
        self.k_proj = nn.ColumnParallelLinear(...)
        self.v_proj = nn.ColumnParallelLinear(...)
        self.o_proj = nn.RowParallelLinearWithReplicateOutput(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias,
        )
```

- `q_proj` / `k_proj` / `v_proj` 使用 **ColumnParallelLinear**：在 tp 维度上按输出列切分权重，每个设备负责部分 attention head
- `o_proj` 使用 **RowParallelLinearWithReplicateOutput**：接受 Attention 输出的 Shard，输出自动 Replicate（mlp 需要 replicate 输入）

**LlamaMLP** — 同样的 ColumnParallel + RowParallel 模式：

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.ColumnParallelLinear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias,
        )
        self.up_proj = nn.ColumnParallelLinear(...)
        self.down_proj = nn.RowParallelLinearWithReplicateOutput(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias,
        )
```

**Embedding** — 使用 `EmbeddingWithReplicateOutput`：

```python
self.embed_tokens = nn.EmbeddingWithReplicateOutput(
    config.vocab_size, config.hidden_size
)
```

### 6.2 测试验证

完整测试代码见 `python/dtorch/test/modules/test_llama.py`。测试覆盖三种场景，均以 HuggingFace Transformers 的 PyTorch 输出为基准，通过 `assert_tensor_allclose` 验证 DTorch 输出与 PyTorch 完全一致：

- **单卡**：不传 `device_mesh`，默认单设备，验证 API 兼容性
- **DP（dp=2）**：`init_device_mesh(device, 2, mesh_dim_names=["dp"])`，验证数据并行
- **TP（tp=2）**：`init_device_mesh(device, 2, mesh_dim_names=["tp"])`，验证张量并行

所有场景下 DTorch 输出与 PyTorch 单卡输出保持一致（`rtol=1e-4, atol=1e-4`），证明分布式切分对计算结果无影响。

---

## 7. 彩蛋：单卡模拟分布式

DTorch 支持在**单张 GPU** 上运行分布式程序（显存足够的前提下）。例如，你可以在只有一张 GPU 的机器上调试 4 卡 TP 并行的代码：

```python
# 在单卡上模拟 dp=2, tp=2 的 4 卡分布式程序
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])
model = LlamaForCausalLM(config, device_mesh=mesh)
# 正常执行，无需多张 GPU！
```

这与 PyTorch 形成鲜明对比：PyTorch 的分布式依赖 NCCL 进行集合通信，而 NCCL 不支持同一张 GPU 上多进程之间的集合通讯，因此 `init_process_group` 要求实际存在 4 张 GPU，单卡环境无法运行多卡分布式代码。DTorch 的 DeviceMesh 是逻辑上的设备拓扑，框架在单卡上自动模拟数据切分和集合通信，让开发者**在单卡上完成分布式程序的开发与调试**，确认无误后再部署到真实多卡集群。

---

## 8. 异步获取 Tensor 值

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

### TensorFuture

`TensorFuture` 提供以下方法：

| 方法 | 说明 |
|---|---|
| `future.get()` | 阻塞等待并返回 `torch.Tensor` |
| `future.wait(timeout_ms)` | 等待最多 `timeout_ms` 毫秒，超时抛出异常 |
| `future.is_ready()` | 非阻塞检查结果是否已 ready（`get()` 消费后返回 `False`） |
| `await future` | 在 asyncio 协程中通过 `await` 异步等待结果（见下方示例） |

```python
future = c.to_torch_async()

# 检查是否 ready（非阻塞）
if future.is_ready():
    result = future.get()

# 带超时等待
try:
    result = future.wait(timeout_ms=5000)
except RuntimeError:
    print("Timed out waiting for tensor value")
```

### await TensorFuture

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

与 `get()` 的区别：
- `future.get()` — 同步阻塞当前线程，适用于普通 Python 代码
- `await future` — 异步等待，释放事件循环给其他协程，适用于 asyncio 程序

### 与 to_torch() 的关系

`to_torch()` 内部实现为 `to_torch_async().get()`，行为完全不变。
对于大多数场景，直接使用 `to_torch()` 即可；在需要异步并发的场景下使用 `to_torch_async()`。

架构细节见 [异步获取 Tensor 值](../developer_guide/eager_graph_architecture/async_get_tensor.md)。

---

## 9. 总结

基于 DTorch 的分布式 API，用户可以**以最小的代码改动**将单机程序拓展为分布式程序，并**保持相同的开发与调试体验**：

- **代码逻辑不变**：算子调用、模型 forward 逻辑与单卡完全一致，仅需在 Tensor 创建时声明 `device_mesh` 和 `placements`
- **自动推导**：框架自动推导输出 Tensor 的分布式信息，无需手动管理通信
- **无需 ProcessGroup**：通过 `redistribute()` 声明式地改变分布策略，框架自动插入集合通讯算子
- **无缝切换**：不传 `device_mesh` 时行为与 PyTorch 完全一致，单卡调试后可直接切换到多卡

核心工作流：

```python
# 1. 定义 DeviceMesh
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])

# 2. 用并行模块替换普通模块（ColumnParallelLinear、RowParallelLinear 等）
# 3. forward 逻辑完全不变
# 4. redistribute_input / redistribute_output 自动处理边界对齐
```

---

## 参考

| 文档 | 内容 |
|---|---|
| [architecture.md](../developer_guide/architecture.md) | 项目整体架构 |
| [distributed_tensor.md](distributed_tensor.md) | DTensor 机制详解 |
| [eager_graph_architecture/operator/placement_signature.md](../developer_guide/operator/placement_signature.md) | PlacementSignature 推导机制 |
| [eager_graph_architecture/operator/operators_mapping.md](../developer_guide/operator/operators_mapping.md) | Python↔C++ 算子映射表 |
| [test.md](../get_started/test.md) | 测试指南 |
