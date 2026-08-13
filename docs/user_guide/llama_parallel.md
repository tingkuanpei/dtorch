# Llama 并行示例：DP + TP + PP + CP

本文以 Llama 模型为例，展示如何使用 DTorch 的 `DeviceMesh` 与 `Module` 体系实现 **Data Parallel、Tensor Parallel、Pipeline Parallel、Context Parallel** 的任意组合。完整代码见 [`python/dtorch/test/modules/llama.py`](https://github.com/tingkuanpei/dtorch/blob/main/python/dtorch/test/modules/llama.py)，测试见 [`python/dtorch/test/modules/test_llama.py`](https://github.com/tingkuanpei/dtorch/blob/main/python/dtorch/test/modules/test_llama.py)。

前置阅读：[Python API 概览](python_api_overview.md)（DTensor 与 `redistribute()`）与 [Module 并行](module_parallel.md)（redistribute 钩子与 Linear 并行子类）。

---

## 1. DeviceMesh：五维命名维度

Llama 使用一个最多五维的 `DeviceMesh`，每类并行对应一个命名维度：

```python
device_mesh = init_device_mesh(
    "cuda",
    (dp, tp, pp, ulysess_cp, ring_cp),
    mesh_dim_names=["dp", "tp", "pp", "ulysess_cp", "ring_cp"],
)
```

| 维度名 | 并行类型 | 作用 |
|---|---|---|
| `dp` | Data Parallel | 按 batch（第 0 维）切分输入，权重完整复制 |
| `tp` | Tensor Parallel | 切分 `Linear` / `Embedding` 权重（见 [Module 并行](module_parallel.md)） |
| `pp` | Pipeline Parallel | 把不同层划分到多个 stage |
| `ulysess_cp` | Context Parallel（Ulysses） | 按序列维切分 Q/K/V，all-to-all 重组 head |
| `ring_cp` | Context Parallel（Ring） | 按序列维切分 Q/K/V，环形 Attention 通信 |

任意维度取 1 即退化为该维度不并行；模型代码对维度组合完全透明，无需为不同策略改写。

---

## 2. 顶层模型：LlamaForCausalLM

顶层模型负责校验 DeviceMesh 维度名、把输入重分布到模型期望的分布，并在输出处恢复原分布：

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config, device_mesh=None):
        super().__init__()
        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        # 只允许声明支持的维度名
        device_mesh.check_all_dim_names_in_set({"dp", "tp", "pp", "ulysess_cp", "ring_cp"})

        self.model = LlamaModel(config, device_mesh)
        # lm_head 位于最后一个流水线 stage
        with Graph.default_graph().device_mesh_guard(self.model.pp_stage_meshes[-1]):
            self.lm_head = nn.ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False,
            )

    def redistribute_input(self, input_ids):
        # 保存输入原始的 mesh/placements，输出时恢复
        self.input_device_mesh = input_ids.device_mesh
        self.input_placement = input_ids.placements

        # 重分布到第一个 stage（embedding）的 mesh：
        #   dp         -> Shard(0)    按 batch 切分
        #   tp         -> Replicate() TP 由权重切分承担，输入无需切分
        #   ulysess_cp / ring_cp -> Shard(1)  按序列维切分
        # 目标 mesh 上不存在的维度名会被自动忽略，因此可以无条件列出。
        input_ids = input_ids.redistribute_by_dict(
            self.first_param_device_mesh(),
            placements_dict={
                "dp": Shard(0),
                "tp": Replicate(),
                "ulysess_cp": Shard(1),
                "ring_cp": Shard(1),
            },
        )
        return [input_ids], {}

    def redistribute_output(self, logits):
        # 恢复为输入的原始分布，保证对外透明
        return logits.redistribute(self.input_device_mesh, placements=self.input_placement)
```

关键点：

- `check_all_dim_names_in_set(...)` 确保 DeviceMesh 的维度名不超出支持范围
- `placements_dict` 中的维度名**不需要提前判断是否存在**：目标 mesh 上没有对应维度时该项被忽略
- `redistribute_input` 把输入转到第一个 stage 的 mesh，`redistribute_output` 从最后一个 stage（lm_head 所在处）恢复为调用方原始的分布

---

## 3. Data Parallel：输入按 batch 切分

与 TP/PP/CP 不同，DP 不涉及任何模型结构——它完全由**数据的分布**体现：输入在 `dp` 维上按 batch 切分为 `Shard(0)`，权重在 `dp` 维上保持 `Replicate()`（完整复制）。

- 输出端在 `redistribute_input` 中 `"dp": Shard(0)` 就是模型里全部与 DP 相关的代码：切分后每个 dp 副本对自己那份 batch 独立执行完整前向，`dp` 维上全程无通信。
- 输出端由 `redistribute_output` 恢复调用方的原始分布：

```python
# 输入端
placements_dict={
    "dp": Shard(0),        # 每个 dp 副本处理 batch 的一份切片
    "tp": Replicate(),
    ...
}

# 输出恢复为调用方分布：Replicate 调用方 -> 沿 dp 维 AllGather
logits = logits.redistribute(self.input_device_mesh, placements=self.input_placement)
```

---

## 4. Pipeline Parallel：层到 stage 的划分

`LlamaModel` 接收完整的 `device_mesh`，在内部用 `unbind("pp")` 展开出每个 stage 的子 mesh，并把每一层绑定到其所属 stage。其中`assign_layers_to_stages(num_layers, num_stages)` 把层尽可能均匀地分给各 stage（无法整除时靠前的 stage 多分一层），例如 `assign_layers_to_stages(4, 2)` 返回 `[0, 0, 1, 1]`。

```python
class LlamaModel(nn.Module):
    def __init__(self, config, device_mesh):
        super().__init__()
        # unbind("pp") 把 pp 维展开为一组子 mesh（去掉 pp 维、其余维度不变）；
        # 无 pp 维时返回 [device_mesh]，退化为单 stage。
        self.pp_stage_meshes = device_mesh.unbind("pp")
        # 把 num_hidden_layers 层均匀映射到各 stage，得到每层所属的 stage 编号
        self.layer_stage_ids = assign_layers_to_stages(
            config.num_hidden_layers, len(self.pp_stage_meshes)
        )

        # embedding 与 rotary 位于第一个 stage
        with Graph.default_graph().device_mesh_guard(self.pp_stage_meshes[0]):
            self.rotary_emb = LlamaRotaryEmbedding(config=config)
            self.embed_tokens = nn.EmbeddingWithReplicateOutput(config.vocab_size, config.hidden_size)

        # 每个 decoder 层绑定到其所属 stage 的 mesh
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            this_stage_device_mesh = self.pp_stage_meshes[self.layer_stage_ids[layer_idx]]
            with Graph.default_graph().device_mesh_guard(this_stage_device_mesh):
                self.layers.append(LlamaDecoderLayer(config, layer_idx))

        # 最终 norm 位于最后一个 stage
        with Graph.default_graph().device_mesh_guard(self.pp_stage_meshes[-1]):
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

`forward` 中按层序执行，进入每层前把激活重分布到该层的 stage mesh，跨 stage 时自动搬运激活：

```python
def forward(self, input_ids=None):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    position_ids = dtorch.arange(
        inputs_embeds.shape[1], device_mesh=hidden_states.device_mesh
    ).unsqueeze(0)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(self.layers):
        this_stage_device_mesh = self.pp_stage_meshes[self.layer_stage_ids[layer_idx]]
        # hidden_states 和 position_embeddings 均 redistribute 到 this_stage_device_mesh
        hidden_states = hidden_states.redistribute(
            device_mesh=this_stage_device_mesh, placements=hidden_states.placements
        )
        position_embeddings = [
            pe.redistribute(device_mesh=this_stage_device_mesh, placements=pe.placements)
            for pe in position_embeddings
        ]
        hidden_states = decoder_layer(
            hidden_states, position_embeddings=position_embeddings,
        )

    hidden_states = hidden_states.redistribute(
        device_mesh=self.pp_stage_meshes[-1], placements=hidden_states.placements
    )
    hidden_states = self.norm(hidden_states)
    return hidden_states
```

---

## 5. Context Parallel：RoPE 与序列切分

与 DP 一样，CP 不切分任何权重，切分完全由数据分布体现——始于 `redistribute_input` 的 `placements_dict`：`input_ids` 在 `ulysess_cp` / `ring_cp` 维上按序列维切分为 `Shard(1)`（`[N, L]` 布局中的 `L` 维）：

```python
placements_dict={
    "dp": Shard(0),
    "tp": Replicate(),
    "ulysess_cp": Shard(1),   # 每个 CP rank 持有序列的一份切片
    "ring_cp": Shard(1),
}
```

序列切分随 `Shard(1)` 在模型中自然传播：embedding 输出的 `hidden_states` 为 `[N, L, E]`（`L` 维切分），经 Q/K/V 投影后 `view(bsz, q_len, -1, head_dim).transpose(1, 2)` 把布局转成 `[N, H, L, E]`，序列维的 `Shard` 随之移动到第 2 维——attention 要求的 Q/K/V `Shard(2)`（即 `[N, H, L, E]` 布局中的 `L` 维）由输入切分与 layout 变换自动得到，无需任何显式 `redistribute`。

只要 mesh 中存在 `ulysess_cp` / `ring_cp` 维度，`scaled_dot_product_attention` 就会自动启用对应的 CP 实现——attention 层无需任何额外代码。

位置编码同样**无需任何额外处理**，当它们与 CP 切分后的 Q/K 相乘时，广播二元算子（`broadcast_op_imlp.cc`）会自动把 `Replicate` 一侧转换为 `Shard`，与 Q/K 的分布对齐。

---

## 6. Tensor Parallel：Attention 与 MLP

TP 复用 [Module 并行](module_parallel.md) 中介绍的 `ColumnParallelLinear` + `RowParallelLinearWithReplicateOutput` 组合，CP/PP 的加入不影响任何权重切分逻辑：

```python
class LlamaSdpaAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        ...
        # ColumnParallel：按输出列（attention head）在 tp 维切分
        self.q_proj = nn.ColumnParallelLinear(self.hidden_size, self.num_heads * self.head_dim, bias=...)
        self.k_proj = nn.ColumnParallelLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=...)
        self.v_proj = nn.ColumnParallelLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=...)
        # RowParallel：按输入维切分，输出自动 AllReduce 为 Replicate
        self.o_proj = nn.RowParallelLinearWithReplicateOutput(self.num_heads * self.head_dim, self.hidden_size, bias=...)

    def forward(self, hidden_states, position_embeddings):
        ...
        # mesh 含 ulysess_cp / ring_cp 维度时，CP 自动启用
        attn_output = dtorch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True,
        )
        ...
```

MLP 与 Embedding 同理：

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.ColumnParallelLinear(self.hidden_size, self.intermediate_size, bias=...)
        self.up_proj   = nn.ColumnParallelLinear(...)
        self.down_proj = nn.RowParallelLinearWithReplicateOutput(self.intermediate_size, self.hidden_size, bias=...)

# Embedding 按 embedding_dim 切分，输出 AllGather 为 Replicate
self.embed_tokens = nn.EmbeddingWithReplicateOutput(config.vocab_size, config.hidden_size)
```

---

## 7. 四个并行维度的分工总结

| 组件 | dp | tp | pp | cp |
|---|---|---|---|---|
| 输入 ids | `Shard(0)`（batch） | `Replicate()` | 位于第一个 stage | `Shard(1)`（序列） |
| `Linear` / `Embedding` 权重 | `Replicate()` | `Shard(0|1)` | 每 stage 持有自己的层 | `Replicate()` |
| Q/K/V | 随输入 | 随 ColumnParallel 输出 | — | `Shard(2)`（序列） |
| cos/sin | `Replicate()` | `Replicate()` | 随激活跨 stage 搬运 | `Replicate()`（广播算子自动转 `Shard`） |

---

## 8. 测试：一键切换并行策略

测试以 PyTorch 单卡模型为参考，通过一个 `dtorch_imp` 闭包任意组合并行策略（参考 `python/dtorch/test/modules/test_llama.py`）：

```python
def _test_llama(test_case, device):
    torch_out = torch_llama(torch_in)[0]   # PyTorch 参考输出

    def dtorch_imp(dp=1, tp=1, pp=1, ulysess_cp=1, ring_cp=1):
        device_mesh = init_device_mesh(
            device,
            (dp, tp, pp, ulysess_cp, ring_cp),
            mesh_dim_names=["dp", "tp", "pp", "ulysess_cp", "ring_cp"],
        )
        if not is_graph_satisfy(dtorch.default_graph, device_mesh):
            return
        dtorch_in = dtorch.Tensor(torch_in)
        dtorch_llama = LlamaForCausalLM(config, device_mesh=device_mesh)
        dtorch_llama.load_state_dict(torch_llama.state_dict())
        dtorch_out = dtorch_llama(dtorch_in)
        assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)

    dtorch_imp(dp=2)
    dtorch_imp(dp=2, tp=2, pp=2, ulysess_cp=2)   # DP+TP+PP+CP 组合
```

模型代码不感知具体策略：任意组合只是改变 `DeviceMesh` 的形状与维度名，`is_graph_satisfy` 在集群不满足 mesh 要求时自动跳过，保证测试在任意规模的环境都能运行。

测试命令如下：没有多卡集群时，也可以开启[单卡模拟分布式](python_api_overview.md#6-彩蛋单卡模拟分布式)在单张 GPU 上运行全部策略组合：

```bash
# `DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE=16` 表示用 16 个虚拟 GPU 模拟（默认 8）
# 需保证 `dp × tp × pp × ulysess_cp × ring_cp ≤ 16`。
DTORCH_DTENSOR_IN_SAME_DEVICE=1 DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE=16 python3 python/dtorch/test/modules/test_llama.py
```
