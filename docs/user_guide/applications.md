# Applications: 扩散模型推理应用

`python/dtorch/applications/` 目录基于 DTorch API 实现扩散模型（Diffusion Models）的分布式推理，目前支持 **Stable Diffusion 3 Medium** 和 **FLUX.1-dev** 两个模型。

---

## 1. 目录结构

```
python/dtorch/applications/
├── __init__.py
├── LICENSE
├── core/                              # DTorch 原生编写的核心工具
│   ├── __init__.py
│   ├── config.py                      # ExecuteConfig 及子配置（缓存、量化、算子融合）
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── cache_base.py              # CacheBase 抽象基类 + enable_cache 工厂函数
│   │   ├── first_block_cache.py       # First Block Cache 残差缓存优化
│   │   └── tea_cache.py               # TeaCache 缓存优化
│   └── util/
│       ├── __init__.py
│       └── image_metrics.py           # 图像质量评估指标
│
├── huggingface/                        # 从 transformers / diffusers 迁移的代码
│   ├── transformers/                  # 从 transformers==4.53.1 迁移
│   └── diffusers/                     # 从 diffusers==0.34.0 迁移
|
└── test/                               # 应用级集成测试
    ├── __init__.py
    └── test_model.py                  # Flux / SD3 端到端测试
```

### 目录划分

| 目录 | 说明 |
|---|---|
| `core/` | DTorch 原生代码：配置系统、缓存优化、性能评估工具 |
| `huggingface/transformers/` | 从 `transformers` 迁移的文本编码器（CLIP、T5） |
| `huggingface/diffusers/` | 从 `diffusers` 迁移的模型与管线（Transformer、VAE、Scheduler、Pipeline） |
| `test/` | 应用级测试，对比 DTorch 分布式输出与 PyTorch 单卡输出的一致性 |

### 代码迁移原则

`huggingface/` 目录中的代码均从 `transformers==4.53.1` 和 `diffusers==0.34.0` 迁移而来，遵循两个原则：

1. **目录结构与文件名与原库一致** — 例如 `diffusers/models/attention_processor.py` → `huggingface/diffusers/models/attention_processor.py`
2. **仅做必要修改** — 只修改为支持 DTorch API 必须修改的内容，以及为了简化实现（避免引入过多代码文件）而做的必要简化

所有迁移文件保留原始库的 Apache 2.0 版权声明，并在下方追加 DTorch 版权声明。

---

## 2. 支持 DTorch API 的必要修改

以下是为了让模型使用 DTorch API 在单机单卡环境下跑起来而需要做的**最基础修改**。分布式、Cache、算子融合等优化项在后续章节单独说明。

### 2.1 模块基类替换

所有模型的基类从 `torch.nn.Module` 替换为 `dtorch.nn.Module`，使得 DTorch 的 `__call__` 钩子（图构建、算子捕获）能够正常介入模型前向过程：

| 原库基类 | DTorch 基类 |
|---|---|
| `diffusers.models.modeling_utils.ModelMixin` | 继承 `dtorch.nn.Module` |
| `transformers.modeling_utils.PreTrainedModel` | 继承 `dtorch.nn.Module` |

### 2.2 导入替换

全面替换 `torch` 导入为 `dtorch` 导入，使所有算子调用经过 DTorch 的算子分发系统：

```python
# 原库
import torch
import torch.nn as nn
import torch.nn.functional as F

# DTorch 迁移后
import dtorch
import dtorch.nn as nn
import dtorch.nn.functional as F
```

### 2.3 Tokenizer 输入包装

Pipeline 中 HuggingFace tokenizer 产出的 `torch.Tensor` 显式包装为 DTorch Tensor，使其能传入 DTorch 模型：

```python
text_input_ids = text_inputs.input_ids           # torch.Tensor
text_input_ids = dtorch.Tensor(text_input_ids)   # dtorch.Tensor
prompt_embeds = self.text_encoder_2(text_input_ids.to(device), ...)
```

### 2.4 dtype_guard

`from_pretrained` 中使用 `dtorch.default_graph.dtype_guard(torch_dtype)` 确保模型权重以指定的 dtype 加载：

```python
with dtorch.default_graph.dtype_guard(torch_dtype):
    model = cls(config, *model_args, **model_kwargs, **kwargs)
```

### 2.5 Scheduler 保持独立

Scheduler 仅做时序计算（基于 numpy），不涉及张量分布式操作。因此 Scheduler 不从 `dtorch.nn.Module` 继承，而是保持轻量的独立类，仅复用来自原库的 `ConfigMixin` 做配置管理。

---

## 3. ExecuteConfig

`ExecuteConfig` 是 DTorch 应用层的统一配置入口，定义在 `core/config.py`，贯穿 Pipeline → Model → Attention Processor 全链路。其结构如下：

```python
@dataclass
class ExecuteConfig:
    cache_config: Optional[Union[FirstBlockCacheConfig, TeaCacheConfig]] = None
    quant_config: QuantizeConfig = QuantizeConfig()
    fuse_kernel_config: FuseKernelConfig = FuseKernelConfig()

@dataclass
class QuantizeConfig:
    sage_attn_type: Optional[str] = None   # Sage Attention 量化模式

@dataclass
class FuseKernelConfig:
    fuse_apply_rotary_emb: bool = False    # 融合 RoPE 到 Attention Kernel
    fuse_silu_linear_chunk: bool = False   # 融合 SiLU + Linear
    fuse_layer_norm_mul_add: bool = False  # 融合 LayerNorm + Mul + Add

@dataclass
class FirstBlockCacheConfig(CacheConfig):
    residual_diff_threshold: float = 0.12
    downsample_factor: int = 1

@dataclass
class TeaCacheConfig(CacheConfig):
    pass
```

### 传递路径

```
test_model.py 构造 ExecuteConfig
    → Pipeline.__init__(execute_config=...)
        → 传给 Transformer 模型 (model.__init__(execute_config=...))
            → 传给 Attention Processor (FluxAttnProcessor2_0(execute_config=...))
        → 传给 Cache (enable_cache(..., execute_config.cache_config, ...))
```

Pipeline 构造时会从 `ExecuteConfig` 中取出 `cache_config` 初始化缓存上下文，其余配置随模型转发到各子模块。

---

## 4. 分布式

### 4.1 DeviceMesh 感知的模型构造

所有模型构造函数接受 `device_mesh: Optional[DeviceMesh]` 参数，在 `device_mesh_guard` 上下文中构造子模块，确保所有 `dtorch.nn` 层在构造时绑定到正确的 DeviceMesh：

```python
def __init__(self, ..., device_mesh: Optional[DeviceMesh] = None, ...):
    device_mesh = get_default_device_mesh(device_mesh=device_mesh)
    device_mesh.check_all_dim_names_in_set({"dp", "tp", "ulysess_cp", "ring_cp"})

    with Graph.default_graph().device_mesh_guard(device_mesh):
        self.pos_embed = PatchEmbed(...)
        self.norm = AdaLayerNormZero(dim)
        # ...
```

### 4.2 Pipeline 的 DeviceMesh 分派

Pipeline 负责为不同组件分配合适粒度的 DeviceMesh：

| 组件 | DeviceMesh | 原因 |
|---|---|---|
| Transformer | 完整 mesh（dp + tp + ulysess_cp + ring_cp） | 扩散模型主干，所有并行维度都参与 |
| Text Encoder | 缩减 mesh（`unbind(["ulysess_cp", "ring_cp"])`） | 文本编码不需要序列并行 |
| VAE / Scheduler | 单设备（`first_device()`） | 不参与分布式计算 |

### 4.3 并行线性层

将普通 `nn.Linear` 替换为 DTensor 感知的并行线性层：

| 原库 | DTorch 替换 | 使用场景 |
|---|---|---|
| `nn.Linear(in, out)` | `nn.ColumnParallelLinear(in, out)` | Q/K/V 投影，沿输出列切分到各设备 |
| `nn.Linear(in, out)` | `nn.RowParallelLinearWithReplicateOutput(in, out)` | 输出投影，沿输入行切分后 AllReduce |
| `nn.Linear(in, out)` | `nn.ColumnParallelLinearWithReplicateInputOutput(in, out)` | 输入输出均需 replicate 的场景 |
| `nn.Linear(in, out)` | `nn.ReplicateParallelLinear(in, out)` | 条件嵌入投影，所有设备持有完整副本 |

### 4.4 Embedding 层

将 `nn.Embedding` 替换为 `nn.EmbeddingWithReplicateOutput`，查表后自动 replicate 输出，避免在分布式环境中产生 shard 不一致：

```python
self.token_embedding = nn.EmbeddingWithReplicateOutput(config.vocab_size, embed_dim)
self.position_embedding = nn.EmbeddingWithReplicateOutput(config.max_position_embeddings, embed_dim)
```

---

## 5. Cache

`core/cache/` 实现了残差缓存优化，通过 hook transformer block 的 forward 过程，在相邻推理步之间复用变化较小的 hidden states，减少计算量。

### 架构

```
CacheBase (ABC)
├── FirstBlockCache    # 已实现
└── TeaCache           # 预留接口，待实现
```

### enable_cache 工厂

```python
def enable_cache(pipe_name: str, cache_config, transformer_blocks) -> CacheBase:
    if cache_config is None:
        return None
    if isinstance(cache_config, FirstBlockCacheConfig):
        return FirstBlockCache(pipe_name, cache_config)
```

### First Block Cache

`FirstBlockCache` 注册 hook 到每个 transformer block 的 forward 过程：

1. 在第一个 transformer block 之后比较当前步与上一步的 hidden states 差异
2. 若差异小于 `residual_diff_threshold`，跳过后续 transformer blocks，直接复用上一步的输出
3. 若差异超过阈值，正常执行全部 blocks 并更新缓存

```python
@dataclass
class FirstBlockCacheConfig(CacheConfig):
    residual_diff_threshold: float = 0.12   # 残差差异阈值
    downsample_factor: int = 1              # 下采样因子
```

---

## 6. 算子融合

算子融合通过 `FuseKernelConfig` 控制，减少 kernel launch 开销和中间张量显存占用。

### 融合选项

| 选项 | 说明 | 使用位置 |
|---|---|---|
| `fuse_apply_rotary_emb` | 将 RoPE 位置编码融合进 Attention Kernel | `FluxAttnProcessor2_0.__call__` |
| `fuse_silu_linear_chunk` | 融合 SiLU 激活 + Linear 投影 | `FluxSingleTransformerBlock.forward` / `FluxTransformerBlock.forward` |
| `fuse_layer_norm_mul_add` | 融合 LayerNorm + 逐元素乘法 + 残差加法 | `AdaLayerNormZero.forward` |

### 使用方式

```python
# RoPE 融合
if self.fuse_apply_rotary_emb:
    query = F._apply_rotary_emb(query, image_rotary_emb)
    key = F._apply_rotary_emb(key, image_rotary_emb)
else:
    query = apply_rotary_emb(query, image_rotary_emb)
    key = apply_rotary_emb(key, image_rotary_emb)

# SiLU + Linear 融合（在 AdaLayerNormZero 中控制）
norm_hidden_states, gate = self.norm(
    hidden_states, emb=temb,
    fuse_silu_linear_chunk=self.fuse_silu_linear_chunk,
    fuse_layer_norm_mul_add=self.fuse_layer_norm_mul_add,
)
```

---

## 7. 量化

量化通过 `QuantizeConfig` 控制，目前仅支持 **Sage Attention** 量化加速。

### Sage Attention

注意力计算中通过 `dtorch.SdpaOption` 传递量化配置，支持 FP8 / INT8 等量化模式加速 Attention 计算：

```python
hidden_states = F.scaled_dot_product_attention(
    query, key, value,
    dropout_p=0.0, is_causal=False,
    sdpa_option=dtorch.SdpaOption(
        sage_attn_type=self.sage_attn_type,  # None / "auto" / "default"
    ),
)
```

配置通过 `QuantizeConfig` → `ExecuteConfig` → `FluxAttnProcessor2_0`（或 `JointAttnProcessor2_0`）传递：

```python
@dataclass
class QuantizeConfig:
    sage_attn_type: Optional[str] = None
```

| sage_attn_type | 说明 |
|---|---|
| `None` | 不启用量化，使用标准 SDPA |
| `"auto"` | 自动选择量化模式 |
| `"default"` | 默认量化模式 |

---

## 8. 其他实现差异

### 8.1 数值精度处理

部分归一化层在计算时内部提升为 float32，防止混合精度下的数值溢出：

```python
class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: dtorch.Tensor) -> dtorch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(),
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps).to(origin_dtype)
```

### 8.2 配置类依赖保留

虽然替换了模型基类，但仍从原库导入配置基础设施，因为它们不会引入额外代码文件，且能保证与原模型权重的配置兼容：

```python
from transformers.configuration_utils import PretrainedConfig
from diffusers.configuration_utils import ConfigMixin, register_to_config
```

### 8.3 from_pretrained 仅支持本地路径

`from_pretrained` 不接入 HuggingFace Hub，仅从本地文件系统加载模型。Pipeline 构造时通过 `model_index.json` 定位各子模型目录：

```python
class DiffusionPipeline:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_path = pretrained_model_name_or_path
        # 从 model_index.json 读取组件映射
        with open(os.path.join(model_path, "model_index.json"), "r") as file:
            model_index_data = json.load(file)
        # 通过 class_mapping 将类名映射到 DTorch 类
        # 分组件调用各自的 from_pretrained
```

### 8.4 Scheduler 时序计算

`FlowMatchEulerDiscreteScheduler` 的时序计算基于 numpy，将 timesteps 转为 `dtorch` 张量仅用于兼容下游，不涉及分布式执行。

---

## 9. 被删除的功能

以下功能在原库中存在，但在迁移版本中被有意删除或未实现：

| 功能 | 说明 |
|---|---|
| **在线模型下载** | `from_pretrained` 不支持 HuggingFace Hub 下载，仅支持本地路径 |
| **LoRA** | `lora_scale` 参数被 assert 拒绝，无 LoRA 权重加载逻辑 |
| **Textual Inversion** | `TextualInversionLoaderMixin` 相关路径被 assert 跳过 |
| **Image Encoder (Flux)** | `CLIPVisionModelWithProjection` 被注释掉 |
| **Attention Mask** | 大多数 Attention Processor assert `attention_mask is None` |
| **upcast_attention** | 不支持 `upcast_attention=True`，Attention 构造时保留参数但未使用 |
| **only_cross_attention** | 不支持仅交叉注意力的 Attention 模式 |
| **rms_norm (AdaLayerNormContinuous)** | `norm_type="rms_norm"` 被 assert 拒绝 |
| **rms_norm_across_heads / l2 等 qk_norm** | qk_norm 仅支持 `layer_norm` / `fp32_layer_norm` / `rms_norm` |
| **online scheduler config** | Scheduler 不从 Hub 下载配置，仅读取本地 `scheduler_config.json` |
| **Karras / Exponential / Beta sigmas** | Scheduler 中这些时序调度策略被 assert 拒绝 |

---

## 10. 参考链接

| 文档 | 说明 |
|---|---|
| [python_api.md](python_api.md) | Python API 使用指南 |
| [architecture.md](../developer_guide/architecture.md) | DTorch 整体架构文档 |

### 外部参考

| 资源 | 说明 |
|---|---|
| [transformers==4.53.1](https://github.com/huggingface/transformers/tree/v4.53.1) | HuggingFace Transformers 源码（文本编码器参考） |
| [diffusers==0.34.0](https://github.com/huggingface/diffusers/tree/v0.34.0) | HuggingFace Diffusers 源码（Pipeline / Transformer / VAE / Scheduler 参考） |
