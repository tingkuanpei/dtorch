# Distributed Tensor (DTensor)

DTensor（Distributed Tensor）是描述 ”如何将一个逻辑 Tensor 切分并分布到多个设备上“ 的抽象，每个设备持有该 Tensor 的一部分数据（或完整副本）。DTorch 原生支持 DTensor，通过 **DeviceMesh** 和 **Placements** 描述其在分布式集群上的分布方式。基于 DTensor，用户可以组合实现 Data Parallel、Tensor Parallel、Context Parallel、Pipeline Parallel、Expert Parallel 和 ZeRO 等所有主流并行方案——只需声明 Tensor 的分布策略，框架自动完成切分、通信和同步，无需手动管理 ProcessGroup 或调用集合通信算子。

## 1. DeviceMesh — 描述设备拓扑

`DeviceMesh` 是一个 N 维设备网格，描述集群中设备的逻辑拓扑结构。

```python
from dtorch import init_device_mesh

# 1D DeviceMesh: 4 个 GPU 排成一条线
mesh_1d = init_device_mesh("cuda", 4)
# DeviceMesh('cuda', dim_name: [], shape: (4,), data: [[0, 1, 2, 3]])

# 2D DeviceMesh: 2×4=8 个 GPU，命名为 ("dp", "tp")
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=["dp", "tp"])
# DeviceMesh('cuda', dim_name: ['dp', 'tp'], shape: (2, 4), data: [[0, 1, 2, 3], [4, 5, 6, 7]])
```

### DeviceMesh 图示

```
1D DeviceMesh (4 GPUs):
┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │
└───┴───┴───┴───┘

2D DeviceMesh (2×4, dim_names=["dp", "tp"]):
       tp →
    ┌───┬───┬───┬───┐
dp  │ 0 │ 1 │ 2 │ 3 │  ← tp 维度：4 个设备
↓   ├───┼───┼───┼───┤
    │ 4 │ 5 │ 6 │ 7 │  ← dp 维度：2 个设备
    └───┴───┴───┴───┘
```

## 2. Placements — 描述 Tensor 分布

每个 DTensor 有一个 `PlacementSeq`，长度等于 DeviceMesh 的维度数。每个维度上可以使用三种 Placement：

| Placement | 含义 | 构造函数 | 图示 |
|-----------|------|----------|------|
| `Replicate()` | 完整复制到该维度所有设备 | `"R"` | 每个设备持有完整数据 |
| `Shard(dim)` | 沿第 `dim` 轴切分到各设备 | `"S{index}"`，如 `"S0"`, `"S1"` | 每设备持有 1/N 数据 |
| `Partial()` | 各设备持有部分结果，需 reduce sum 得到完整值 | `"P"` | 每设备持有部分和 |

```python
from dtorch import Replicate, Shard, Partial

# Shard: 沿指定维度切分
s0 = Shard(0)   # 沿第 0 维切分 → "S0"
s1 = Shard(1)   # 沿第 1 维切分 → "S1"

# Replicate: 完整复制
r = Replicate()  # "R"

# Partial: 部分结果（如矩阵乘法中间结果）
p = Partial()    # "P"
```

### Placement 图示

以一个形状为 `[4, 8]` 的 2D Tensor 分布在 4 个设备上的 `DeviceMesh("cuda", 4)` 为例：

```
原始 Tensor (4×8):
┌────────────────────────────────┐
│ a₀₀ a₀₁ ... a₀₇ │  ← 第 0 行
│ a₁₀ a₁₁ ... a₁₇ │  ← 第 1 行
│ a₂₀ a₂₁ ... a₂₇ │  ← 第 2 行
│ a₃₀ a₃₁ ... a₃₇ │  ← 第 3 行
└────────────────────────────────┘

Placements=[Shard(0)]: 沿第 0 维切分，每设备 1 行
 GPU 0      GPU 1      GPU 2      GPU 3
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ 第0行  │ │ 第1行  │ │ 第2行  │ │ 第3行  │
└────────┘ └────────┘ └────────┘ └────────┘

Placements=[Shard(1)]: 沿第 1 维切分，每设备 2 列
 GPU 0      GPU 1      GPU 2      GPU 3
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ 第0-1列│ │ 第2-3列│ │ 第4-5列│ │ 第6-7列│
│  (4行) │ │  (4行) │ │  (4行) │ │  (4行) │
└────────┘ └────────┘ └────────┘ └────────┘

Placements=[Replicate()]: 每设备持有完整数据
 GPU 0      GPU 1      GPU 2      GPU 3
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ 完整   │ │ 完整   │ │ 完整   │ │ 完整   │
│ 4×8    │ │ 4×8    │ │ 4×8    │ │ 4×8    │
└────────┘ └────────┘ └────────┘ └────────┘
```

### 多维 DeviceMesh 和 Placements

DeviceMesh 的维度数 N 决定了 `PlacementSeq` 的长度必须为 N — **每个 DeviceMesh 维度对应一个 Placement**，逐维描述 Tensor 在该设备维度上的分布方式。例如一个 2D DeviceMesh `(2, 4)`（dim_names=`["dp", "tp"]`）需要两个 Placement：

```python
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=["dp", "tp"])

# PlacementSeq=[Shard(0), Replicate()]
#   dp 维度 (2 个设备): Shard(0) → 沿第 0 维切分，每组 dp 设备各持有 1/2 数据
#   tp 维度 (4 个设备): Replicate() → 每个 tp 设备持有完整数据
x = dtorch.randn(8, 128, device_mesh=mesh_2d, placements=[Shard(0), Replicate()])
```

```
PlacementSeq 与 2D DeviceMesh 的对应关系:
  mesh_2d = init_device_mesh("cuda", (2, 4), dim_names=["dp", "tp"])
  placements = [Shard(0),  Replicate()]
                ↑          ↑
                dp 维度    tp 维度
                (2 设备)    (4 设备)

  dp 维度 (2 个设备): Shard(0) — 每个 dp 组持有 Tensor 的一半行
  ┌──────────────────────────────────────┐
  │  dp=0: 持有 data[0:4, :]   (4×128)   │
  │  dp=1: 持有 data[4:8, :]   (4×128)   │
  └──────────────────────────────────────┘

  tp 维度 (4 个设备): Replicate() — 每个 tp 设备持有完整数据
  ┌──────────────────────────────────────────────┐
  │  tp=0..3: 每设备持有完整的 4×128 或 4×128    │
  └──────────────────────────────────────────────┘
```

借助这种机制，PlacementSeq 可以描述任意 N 维并行策略。例如 DP+TP 组合时，`placements=[Shard(0), Shard(1)]` 表示 dp 维度沿 batch 切分、tp 维度沿 hidden 维度切分，框架自动处理两维度的协同通信。


## 3. 创建 DTensor

DTorch 提供与 PyTorch 一致的 Tensor 创建接口，额外接受 `device_mesh` 和 `placements` 参数：

```python
import torch
import dtorch
from dtorch import Shard, Replicate

# --- 方式 1: 从 torch.Tensor 创建分布式 Tensor ---
torch_x = torch.randn(8, 128)
device_mesh = dtorch.DeviceMesh("cpu", [0, 1, 2, 3])

# 第 0 维切分到 4 个设备
x_shard = dtorch.from_torch(torch_x, device_mesh=device_mesh, placements=[Shard(0)])
print(x_shard.placements)   # [Shard(0)]
print(x_shard.device_mesh)  # DeviceMesh('cpu', shape: (4,), data: [0, 1, 2, 3])

# --- 方式 2: 使用 dtorch 工厂函数 ---
x_rand = dtorch.randn(4, 8, device_mesh=device_mesh, placements=[Shard(1)])
x_zeros = dtorch.zeros(4, 8, device_mesh=device_mesh, placements=[Replicate()])
x_ones = dtorch.ones(4, 8, device_mesh=device_mesh)  # 默认 Replicate

# --- 方式 3: 使用 dtorch.Tensor 构造 ---
x = dtorch.Tensor(torch_x, device_mesh=device_mesh, placements=[Shard(0)])

# --- 方式 4: 2D DeviceMesh + 2D Placements ---
mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=["dp", "tp"])
x_2d = dtorch.randn(8, 256, device_mesh=mesh_2d, placements=[Shard(0), Shard(1)])
# dp 维度: Shard(0), tp 维度: Shard(1)
```

## 4. 改变 DTensor 分布 — redistribute

`tensor.redistribute()` 可以在不同 DeviceMesh 和 Placement 之间转换 Tensor 的分布方式，框架自动完成数据通信：

```python
# 创建切分在第 0 维的 Tensor
device_mesh = dtorch.DeviceMesh("cpu", [0, 1])
x_s0 = dtorch.randn(8, 128, device_mesh=device_mesh, placements=[Shard(0)])

# redistribute 到 Replicate（底层执行 all-gather）
x_r = x_s0.redistribute(device_mesh=device_mesh, placements=[Replicate()])

# redistribute 到 Shard(1)（底层执行 all-to-all）
x_s1 = x_s0.redistribute(device_mesh=device_mesh, placements=[Shard(1)])

# redistribute 到另一个 Tensor 的分布
x_aligned = x_s0.redistribute_like(x_r)  # x_aligned 现在也是 Replicate

# 使用 dim_name 指定分布（适用于有命名的 DeviceMesh）
mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=["dp", "tp"])
x = dtorch.randn(8, 256, device_mesh=mesh_2d, placements=[Shard(0), Shard(1)])
x_new = x.redistribute_by_dict(
    placements_dict={"dp": Replicate(), "tp": Shard(0)}
)
```

## 5. Operator 如何处理 DTensor

DTorch 所有 Operator 均原生支持 DTensor，用户无需区分单机还是分布式 Tensor，直接调用即可。

**PlacementSignature — 自动推导输出 Placement**

每个 Operator 内置 **PlacementSignature** 规则表，声明输入与输出 Placement 的映射关系。框架根据输入 DTensor 的实际 Placements 自动匹配规则并推导输出 Placements（此过程**仅推导元信息，不涉及实际通信**）。

**核心原则：不自动注入 redistribute**

> 当输入 Placements 无法匹配签名规则时，DTorch **默认抛出错误**，要求用户调整代码。

设计初衷：`tensor.redistribute()` 是非常重的操作（底层涉及 all-gather、all-to-all、all-reduce 等集合通信），若框架隐式执行，用户将失去对通信开销的感知和优化空间。因此用户需要**关心每个 Tensor 的 Placements 和 DeviceMesh，在必要时显式调用 `tensor.redistribute()`**。

**例外：少数算子自动注入通信**

为兼顾代码简洁性，少数算子会**自动插入 redistribute** 以简化常见场景。例如 `BroadcastBinaryOp`（加减乘除等二元运算）中，当 Replicate Tensor 与 Shard Tensor 进行运算时，框架自动将 Replicate 转为 Shard（实现见 `dtorch/api/cpp/functional/implement/broadcast_op_imlp.cc` 中的 `PlacementR2S`）。这些自动注入通信的算子将在文档中着重说明。

> 更多细节请参考 [`docs/developer_guide/operator/placement_signature.md`](../developer_guide/operator/placement_signature.md)。

## 6. 完整示例

### 6.1 Data Parallel：Replicate 输入，Shard 梯度

```python
import dtorch
from dtorch import Replicate, Shard

# 2 卡 Data Parallel
mesh = dtorch.DeviceMesh("cuda", [0, 1])

# 模型参数 Replicate（每卡一份完整副本）
weight = dtorch.randn(128, 256, device_mesh=mesh, placements=[Replicate()])

# 输入数据 Shard(0)（每卡一半 batch）
data = dtorch.randn(32, 128, device_mesh=mesh, placements=[Shard(0)])

# 直接计算，框架自动处理分布式
output = data @ weight  # 自动匹配 PlacementSignature
print(output.shape)     # [32, 256]，每卡持有 Shard(0) 的输出
```

```
Data Parallel 示意图 (2 GPUs, placements=[Shard(0)]):
  GPU 0                    GPU 1
┌──────────────┐        ┌──────────────┐
│  data[0:16]  │        │  data[16:32] │  ← Shard(0): 每卡一半 batch
│  weight (完整)│        │  weight (完整)│  ← Replicate: 每卡完整权重
│     ↓ @      │        │     ↓ @      │
│ output[0:16] │        │ output[16:32]│  ← 输出继承 Shard(0)
└──────────────┘        └──────────────┘
```

### 6.2 Tensor Parallel：切分权重矩阵

```python
# 2 卡 Tensor Parallel（2D DeviceMesh: dp×tp = 1×2）
mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=["dp", "tp"])

# 权重 Shard(0) 到 tp 维度
weight = dtorch.randn(256, 512, device_mesh=mesh, placements=[Replicate(), Shard(0)])
# dp: Replicate, tp: Shard(0) → 每 tp 设备持有权重的一半行

# 输入 Shard(1) 到 tp 维度
data = dtorch.randn(32, 256, device_mesh=mesh, placements=[Replicate(), Shard(1)])
# dp: Replicate, tp: Shard(1) → 每 tp 设备持有输入的一半列

# 矩阵乘法：Shard(1) × Shard(0) → Partial
output = data @ weight
print(output.placements)  # [Replicate(), Partial()] — 需要 redistribute 还原
```

```
Tensor Parallel 图示 (tp=2):
           weight                           data
    Shard(0) on tp dim               Shard(1) on tp dim
  GPU 0          GPU 1           GPU 0          GPU 1
┌─────────┐   ┌─────────┐     ┌─────────┐   ┌─────────┐
│ w[0:128]│   │w[128:256]│     │d[:,0:128]│   │d[:,128:256]│
└─────────┘   └─────────┘     └─────────┘   └─────────┘
       ↘    data @ weight    ↙
          ┌──────────────┐
          │   Partial()  │  ← 需要 all-reduce 得到完整结果
          └──────────────┘
```

### 6.3 多并行组合：Data Parallel + Tensor Parallel

```python
# 2D DeviceMesh: dp=2, tp=2
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])

# 权重：dp 维度 Replicate，tp 维度 Shard(0) — 经典 TP 权重切分
weight = dtorch.randn(256, 512, device_mesh=mesh, placements=[Replicate(), Shard(0)])

# 输入：dp 维度 Shard(0)（数据并行），tp 维度 Replicate（TP 需要完整输入）
data = dtorch.randn(32, 256, device_mesh=mesh, placements=[Shard(0), Replicate()])

# 直接调用 matmul，PlacementSignature 自动推导：
# Shard(0) + Replicate() → Shard(0)  (dp 维度 batch shard 传递)
# Replicate() + Shard(0) → Partial() (tp 维度产生 partial)
output = data @ weight
print(output.placements)  # [Shard(0), Partial()]

# redistribute 将 Partial 还原为 Replicate（底层执行 all-reduce）
output = output.redistribute_by_dict(
    placements_dict={"tp": Replicate()}
)
print(output.placements)  # [Shard(0), Replicate()]
```

```
DP+TP 组合示意图 (dp=2, tp=2):
         tp=0                tp=1
    ┌──────────┐        ┌──────────┐
dp=0│  GPU 0   │        │  GPU 1   │  ← dp=0 组: data batch[0:16]
    │ w[0:128] │        │w[128:256]│
    └──────────┘        └──────────┘
    ┌──────────┐        ┌──────────┐
dp=1│  GPU 2   │        │  GPU 3   │  ← dp=1 组: data batch[16:32]
    │ w[0:128] │        │w[128:256]│
    └──────────┘        └──────────┘

每个 GPU 独立计算 → dp 维度输出 Shard(0)，tp 维度输出 Partial()
```

### 6.4 使用 redistributed module 简化 TP/PP

DTorch 提供了 `CPTPLinear`、`CPTPEmbedding` 等内置 module，自动处理输入/输出的 redistribute：

```python
import dtorch.nn as nn

mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])

# 使用 TP Linear，自动处理 redistribute_input/redistribute_output
linear = nn.CPTPLinear(256, 512, device_mesh=mesh)
x = dtorch.randn(32, 256, device_mesh=mesh, placements=[Shard(0), Replicate()])
y = linear(x)  # 自动完成输入→输出 Placement 转换
```

## 7. 不均匀切分

DTorch 原生支持 Tensor 的不均匀切分——当 Tensor 某个维度的长度不能被设备数整除时，框架**自动计算每个设备的本地 Shape**（而非报错或要求用户手动调整）。

```python
device_mesh = dtorch.DeviceMesh("cpu", [0, 1, 2, 3])

# 形状 [4, 11] 在第 1 维切分到 4 个设备：11 % 4 ≠ 0，为不均匀切分
x = dtorch.randn(4, 11, device_mesh=device_mesh, placements=[Shard(1)])
# GPU 0: 本地 shape [4, 3]  (多 1 列)
# GPU 1: 本地 shape [4, 3]
# GPU 2: 本地 shape [4, 3]
# GPU 3: 本地 shape [4, 2]  (少 1 列)
```

```
不均匀切分示意 ([4, 11] Shard(1) → 4 设备):
     dim=1: 11 列 / 4 设备
     ┌──────┬──────┬──────┬─────┐
     │ 3 列 │ 3 列  │ 3 列 │ 2 列 │
     │GPU 0 │GPU 1 │GPU 2 │GPU 3│
     └──────┴──────┴──────┴─────┘
```

**redistribute 时的自动 padding**

NCCL 等集合通信库要求所有 rank 的输入 Tensor shape 完全一致。当执行 `tensor.redistribute()` 时，DTorch 会自动为不均匀切分的 Tensor **插入 padding 使其对齐，通信完成后再移除 padding**，整个过程对用户透明。例如将上述 `[Shard(1)]` 的 Tensor redistribute 到 `[Replicate()]` 时，框架会自动补齐 GPU 3 的缺口后再执行 all-gather。

```python
# redistribute 自动处理 padding/unpadding
x_r = x.redistribute(device_mesh=device_mesh, placements=[Replicate()])
# 用户无需感知内部的 padding 逻辑
```

实现细节见 `dtorch/api/cpp/distributed_spec.cc` 中的 `DistributedSpec::ComputeLocalShape()` 和 `dtorch/core/communication/thread_group/thread_group.cc` 中的通信 kernel。

## 8. 特性总结

| 特性 | 说明 |
|------|------|
| **N-D 并行** | DeviceMesh 支持任意维度，PlacementSeq 逐维描述分布 |
| **自动化通信** | redistribute() 自动执行 all-gather、all-to-all、all-reduce 等操作 |
| **不均匀切分** | 支持 Tensor 沿任意维度的不均匀切分，框架自动 padding/去除 padding |
| **命名维度** | DeviceMesh 支持 dim_names（如 "dp"、"tp"），方便按名称管理分布策略 |
| **并行组合** | 同一份代码可组合 DP、TP、PP、CP 等多种并行策略 |
| **隐式 state_dict** | 加载模型时无需手动切分参数，框架根据 DeviceMesh 和 Placements 自动处理 |
