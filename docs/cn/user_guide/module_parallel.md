# Module 并行：实现 DP/TP/CP/PP 组合

DTorch 的 `Module` 体系在接口与用法上与 PyTorch 完全一致——用户编写的模型代码无需任何改动即可在 DTorch 中运行；在此基础上，为了支持分布式，DTorch 为 `Module`（`nn.Module` 子类）增加了少量扩展能力，在保持单卡写法的同时原生支持 **Data Parallel、Tensor Parallel、Context Parallel、Pipeline Parallel** 等的组合。本文以 `Linear` 为例说明 Module 层的并行机制；Llama 模型的完整 DP + TP + PP + CP 示例见 [Llama 并行示例](llama_parallel.md)。

前置阅读：[Python API 概览](python_api_overview.md) 的 DTensor 与 `redistribute()` 章节。

---

## 1. Module 的 redistribute 钩子

`Module` 基类提供了 `redistribute_input()` 和 `redistribute_output()` 钩子，在 `forward` 执行前后自动调用。子类重写这两个方法即可实现**透明的输入/输出重分布**——这是后续 `Linear` 子类与完整模型（Llama）构建并行逻辑的统一机制。

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

典型用法：在 `redistribute_input` 中把输入转到模型期望的分布并保存原始分布，在 `redistribute_output` 中把输出恢复为原始分布，从而对调用方保持透明。

---

## 2. DP / TP / CP / PP 并行实现

DTorch 通过 `DeviceMesh` 的命名维度统一表达各类并行策略——为每个维度赋予语义化名字（`"dp"`、`"tp"`、`"cp"`、`"pp"`），并在 Tensor 和 Parameter 的 `Placements` 中声明各维度上的分布方式，框架据此自动插入集合通信。下面分别说明四类并行在 Module 层面的使用方式。

### Data Parallel

数据并行在 `"dp"` 维度上按 batch 切分输入，权重在 `"dp"` 维度上保持 `Replicate()`。只需在 `DeviceMesh` 中声明一个名为 `"dp"` 的维度，并在模型入口把输入按 batch 切分到该维度：

```python
device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=["dp"])

# 将输入按 batch 切分到 dp 维
input = input.redistribute_by_dict(device_mesh, placements_dict={"dp": Shard(0)})
```

权重在所有非 `"tp"` 维度上默认即为 `Replicate()`（详见第 4 节 Linear 实现解析），因此 DP 无需额外切分，保持各设备权重一致。

### Tensor Parallel

张量并行在 `"tp"` 维度上切分权重。**实际需要切分的只有两类层——`Linear` 与 `Embedding`**：前者通过 `ColumnParallelLinear` / `RowParallelLinear` 子类，后者通过 `EmbeddingWithReplicateOutput` 实现，这些子类均已内置好权重的切分方式与输入输出的校验/转换。

```python
device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=["tp"])

fc1 = nn.ColumnParallelLinear(hidden_size, intermediate_size)                 # 权重在 tp 维上 Shard(0)
fc2 = nn.RowParallelLinearWithReplicateOutput(intermediate_size, hidden_size) # 输出自动 AllReduce 为 Replicate

embed = nn.EmbeddingWithReplicateOutput(vocab_size, hidden_size)              # 按 embedding_dim 切分，输出 AllGather

# 调用方式与普通 nn.Linear / nn.Embedding 一致（token_ids 需在 tp 维上为 Replicate）
h = embed(token_ids)  # -> Replicate（已 AllGather）
h = fc1(h)            # ColumnParallel：Replicate 进，按特征维 Shard 出
h = fc2(h)            # RowParallel：Shard 进，AllReduce 后 Replicate 出
```

- `Linear`：`ColumnParallelLinear` 按输出维 `Shard(0)` 切分；`RowParallelLinearWithReplicateOutput` 按输入维 `Shard(1)` 切分，并在输出处 AllReduce 聚合为 `Replicate`。
- `Embedding`：权重按 `embedding_dim`（`Shard(1)`）切分，`EmbeddingWithReplicateOutput` 在输出处 AllGather，使结果在 tp 维上恢复为 `Replicate`。

> 与 Megatron-LM 不同，DTorch 无需按 rank 手动切分权重并分别加载对应分片：声明 `Placements` 后，DTensor 会在内部自动完成 Tensor 的加载与按维度切分。

### Context Parallel

Context Parallel 在 `"cp"` 维度上按**序列长度**切分 Q/K/V，专门用于长序列 Attention。DTorch 支持两种 CP 变体，由 `DeviceMesh` 的维度名区分：

- **Ulysses CP**（维度名 `"ulysess_cp"`）：按 attention head 切分，内部以 all-to-all 重组。
- **Ring CP**（维度名 `"ring_cp"`）：在序列维上做环形 Attention 通信。

两种变体都要求 Q/K/V 在 CP 维度上为 `Shard(2)`（即 `[N, H, L, E]` 布局中的序列维 `L`）。只要 `DeviceMesh` 中存在 `ulysess_cp` / `ring_cp` 维度，`scaled_dot_product_attention` 即会据此自动启用对应的 CP 实现：

```python
import dtorch
import dtorch.nn.functional as F
from dtorch import init_device_mesh, Shard

device_mesh = init_device_mesh(
    "cuda", (dp, ulysess_cp, ring_cp),
    mesh_dim_names=["dp", "ulysess_cp", "ring_cp"],
)

# Q/K/V 在 cp 维上 Shard(2)：按序列维 L 切分；dp 维按 batch 切分 Shard(0)
placements = [Shard(0), Shard(2), Shard(2)]
query = dtorch.randn(N, H, L, E, device_mesh=device_mesh, placements=placements)
key   = dtorch.randn(N, H, S, E, device_mesh=device_mesh, placements=placements)
value = dtorch.randn(N, H, S, E, device_mesh=device_mesh, placements=placements)

# DeviceMesh 含 ulysess_cp / ring_cp 维度时，CP 自动启用
out = F.scaled_dot_product_attention(query, key, value, is_causal=True)

print(out.device_mesh)   # DeviceMesh('cuda', dim_name: ['dp', 'ulysess_cp', 'ring_cp'], shape: (2, 2, 2), data: ...)
print(out.placements)    # [Shard(0), Shard(2), Shard(2)]
```

> 实现细节见 `python/dtorch/nn/scaled_dot_product_attention_with_cp.py`。

### Pipeline Parallel

流水线并行（Pipeline Parallel）将模型的不同层划分到多个 stage（设备）上，相邻 stage 之间通过 `redistribute` 传递激活。DTorch 在 Module 层面原生支持 PP——只需把每个子 Module 绑定到其所属 stage 的 `DeviceMesh`，即可保持单一的模型定义，无需手动拆分/裁剪模型。

核心是三个工具：

- `device_mesh.unbind("pp")`：将 `"pp"` 维度展开为若干子 `DeviceMesh`，每个对应一个 stage（即去掉 `"pp"` 维、其余维度保持不变的子 mesh；例如 `dp×tp×pp` 的 mesh 会得到一组 `dp×tp` 的 stage mesh）。当 DeviceMesh 不含 `"pp"` 维时，返回 `[device_mesh]`，退化为单 stage。
- `assign_layers_to_stages(num_layers, num_stages)`：把 `num_layers` 层均匀映射到 `num_stages` 个 stage，返回长度为 `num_layers` 的列表，其第 *i* 项即第 *i* 层所属的 stage 编号（无法整除时靠前的 stage 多分一层）。
- `Graph.default_graph().device_mesh_guard(stage_mesh)`：上下文管理器，把在其内创建的子 Module 绑定到指定 stage 的设备。

```python
import dtorch
from dtorch import nn, Graph, DeviceMesh, init_device_mesh, assign_layers_to_stages

class Transformer(nn.Module):
    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        num_layers = 4

        # 1. 展开 pp 维度，得到每个 stage 的 DeviceMesh（去掉 "pp" 维后的子 mesh）
        self.pp_stage_meshes = device_mesh.unbind("pp")
        pp_stages = len(self.pp_stage_meshes)
        # 2. 把每一层均匀映射到某个 stage，得到每层所属的 stage 编号
        self.layer_stage_ids = assign_layers_to_stages(num_layers, pp_stages)

        # 3. 将每个子 Module 绑定到其所属 stage 的 DeviceMesh
        with Graph.default_graph().device_mesh_guard(self.pp_stage_meshes[0]):
            self.tok_embeddings = nn.EmbeddingWithReplicateOutput(vocab_size, hidden_size)

        self.layers = nn.ModuleList()
        for layer_id in range(num_layers):
            layer_device_mesh = self.pp_stage_meshes[self.layer_stage_ids[layer_id]]
            with Graph.default_graph().device_mesh_guard(layer_device_mesh):
                self.layers.append(TransformerBlock(...))

        with Graph.default_graph().device_mesh_guard(self.pp_stage_meshes[-1]):
            self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: dtorch.Tensor):
        h = self.tok_embeddings(tokens)

        for layer_id, layer in enumerate(self.layers):
            layer_device_mesh = self.pp_stage_meshes[self.layer_stage_ids[layer_id]]
            h = h.redistribute(device_mesh=layer_device_mesh)   # 跨 stage 时自动搬运激活
            h = layer(h, self.freqs_cis)

        output = self.output(h).float()
        return output

device_mesh = init_device_mesh("cuda", (dp, tp, pp), mesh_dim_names=["dp", "tp", "pp"])
model = Transformer(device_mesh)
x = dtorch.randn(batch_size, in_dim, device="cuda")
y = model(x)
```

`forward` 中 `h.redistribute(device_mesh=layer_device_mesh)` 负责在相邻 stage 之间搬运激活；当目标层与当前层位于同一 stage 时，该操作不产生实际通信。

---

## 3. Linear 实现解析

DTorch 的 `Linear` 模块（[`源码`](https://github.com/tingkuanpei/dtorch/blob/main/python/dtorch/nn/modules/linear.py)）原生支持 DP、TP、CP 等多种并行策略。其核心原则是：**只有 TP 维度需要切分 Weight，DP 和 CP 维度上 Weight 始终保持完整复制（`Replicate()`）**。

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

其中最常用的是 `ColumnParallelLinear` + `RowParallelLinearWithReplicateOutput` 组合（见 [Llama 并行示例](llama_parallel.md)）。
