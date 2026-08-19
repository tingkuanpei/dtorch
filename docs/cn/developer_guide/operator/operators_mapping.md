# Operators Mapping

DTorch 中算子涉及三个层次的映射关系：

1. **Python 接口**（如 `dtorch.add`、`dtorch.nn.functional.conv2d`）
2. **C++ API 接口**（`dtorch::api::cpp::functional` 命名空间下的函数）
3. **C++ 核心算子**（`dtorch::core::Operator` 的派生类）

映射流程见"源码解读"，最终结果见"映射关系表"。

---

## 源码解读

### 1. Python 接口 ← C++ API 接口

C++ API 接口通过 nanobind **一对一绑定**到 `dtorch._dtorch_py_api.nn.functional`，绑定过程中接口名会做大小写转换以符合 Python 命名规范（如 `Relu` → `relu`、`_Add` → `_add`），再绑定到 `dtorch.nn.functional` 模块。

根据接口名是否以 `_` 开头，最终分别映射到两个模块：

| 接口名特征 | 目标模块 | 示例 | 备注 |
|---|---|---|---|
| 以 `_` 开头 | `dtorch` | `dtorch.matmul` | 由 `@python/dtorch/__init__.py` 导入并去掉前导下划线 |
| 少部分以 `_` 开头 | `dtorch.nn.functional` | `dtorch.nn.functional._contiguous` | 在 `dtorch.Tensor` 或其他内部位置被调用 |
| 不以 `_` 开头 | `dtorch.nn.functional` | `dtorch.nn.functional.conv2d` | 直接在 `dtorch.nn.functional` 中使用 |

这一机制确保了 DTorch 提供与 PyTorch 一致的 API。

### 2. C++ API 接口 ← C++ 核心算子

所有 C++ API 接口定义在 `@dtorch/api/cpp/functional/` 目录下（不含 `implement` 子目录）。

每个 C++ 核心算子 `dtorch::core::Operator` 可派生出一个或多个 C++ API 接口。`@dtorch/api/cpp/functional/*.cc` 中的代码负责构造对应的 `dtorch::core::Operator` 实例：

```cpp
// dtorch/api/cpp/functional/activation.cc

// Relu 接口创建 ActivationParam，通过工厂模式生成 ActivationOp
// ActivationParam 和 ActivationOp 定义在 @dtorch/core/operators/standard/activation_op.h
Tensor ActivationOpImpl(const Tensor& input, core::ActivationType activationType, bool inplace = false,
                        double alpha = 0.0f, double beta = 0.0f, const std::string& approximate = "") {
    std::unique_ptr<core::OpParam> param(new core::ActivationParam(activationType, inplace, alpha, beta, approximate));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor Relu(const Tensor& input, bool inplace) { return ActivationOpImpl(input, core::ActivationType::kReLU, inplace); }
```

`dtorch::core::Operator` 是所有算子的抽象基类。所有核心算子定义在 `@dtorch/core/operators` 目录中。派生类通过工厂模式创建，因此也可以在 `@dtorch/core/operators/operator_factory.cc` 的 `OperatorFactory` 构造函数中找到所有算子的完整列表。

### 3. 示例

以 `dtorch.add` 为例，完整映射链路：

```
dtorch.add → dtorch::api::cpp::functional::_Add → dtorch::core::BroadcastBinaryOp
```

---

## 映射关系表

以下按核心算子汇总其对应的 C++ API 接口与 Python 接口。同一核心算子对应多个接口时，在对应单元格内换行列示。

### 1. 常规算子

定义在 `@dtorch/core/operators/standard` 目录。

| C++ 核心算子 | C++ API 接口 | Python 接口 |
|---|---|---|
| `dtorch::core::ActivationOp` | `Relu` | `dtorch.nn.functional.relu` |
| （同上） | `Sigmoid` | `dtorch.nn.functional.sigmoid` |
| （同上） | `LeakyRelu` | `dtorch.nn.functional.leaky_relu` |
| （同上） | `Elu` | `dtorch.nn.functional.elu` |
| （同上） | `Gelu` | `dtorch.nn.functional.gelu` |
| （同上） | `Silu` | `dtorch.nn.functional.silu` |
| `dtorch::core::BaseMathOp` | `_Exp` | `dtorch.exp` |
| （同上） | `_Square` | `dtorch.square` |
| （同上） | `_Rsqrt` | `dtorch.rsqrt` |
| （同上） | `_Abs` | `dtorch.abs` |
| （同上） | `_Round` | `dtorch.round` |
| （同上） | `_Floor` | `dtorch.floor` |
| （同上） | `_Cos` | `dtorch.cos` |
| （同上） | `_Sin` | `dtorch.sin` |
| （同上） | `_Asin` | `dtorch.asin` |
| （同上） | `Tanh` | `dtorch.nn.functional.tanh` |
| （同上） | `_Neg` | `dtorch.neg` |
| （同上） | `_Reciprocal` | `dtorch.reciprocal` |
| （同上） | `_Log` | `dtorch.log` |
| （同上） | `_Log2` | `dtorch.log2` |
| （同上） | `_Log10` | `dtorch.log10` |
| （同上） | `_Isinf` | `dtorch.isinf` |
| （同上） | `_Isnan` | `dtorch.isnan` |
| `dtorch::core::BatchNormOp` | `BatchNorm` | `dtorch.nn.functional.batch_norm` |
| `dtorch::core::BroadcastBinaryOp` | `_Add` | `dtorch.add` |
| （同上） | `_Sub` | `dtorch.sub` |
| （同上） | `_Mul` | `dtorch.mul` |
| （同上） | `_Div` | `dtorch.div` |
| （同上） | `_Equal` | `dtorch.equal` / `dtorch.eq` |
| （同上） | `_Greater` | `dtorch.greater` / `dtorch.gt` |
| （同上） | `_GreaterEqual` | `dtorch.greater_equal` / `dtorch.ge` |
| （同上） | `_Less` | `dtorch.less` / `dtorch.lt` |
| （同上） | `_LessEqual` | `dtorch.less_equal` / `dtorch.le` |
| （同上） | `_Pow` | `dtorch.pow` |
| （同上） | `_LogicalAnd` | `dtorch.logical_and` |
| （同上） | `_LogicalOr` | `dtorch.logical_or` |
| （同上） | `_Minimum` | `dtorch.minimum` |
| （同上） | `_Maximum` | `dtorch.maximum` |
| `dtorch::core::ChunkOp` | `_Chunk` | `dtorch.chunk` |
| `dtorch::core::ClampOp` | `_Clamp` | `dtorch.clamp` / `dtorch.clip` |
| `dtorch::core::ConcatOp` | `_Concat` | `dtorch.cat` / `dtorch.concat` |
| `dtorch::core::ContiguousOp` | `_Contiguous` | `dtorch.nn.functional._contiguous` |
| `dtorch::core::ConvertOp` | `_To` | `dtorch.nn.functional._to` |
| （同上） | `_Redistribute` | `dtorch.nn.functional._redistribute` |
| `dtorch::core::ConvOp` | `Conv2d` | `dtorch.nn.functional.conv2d` |
| `dtorch::core::CopyOp` | `_Copy` | `dtorch.nn.functional._copy` |
| `dtorch::core::CreateOp` | `_Empty` | `dtorch.empty` |
| （同上） | `_Zeros` | `dtorch.zeros` |
| （同上） | `_Ones` | `dtorch.ones` |
| （同上） | `_Rand` | `dtorch.rand` |
| （同上） | `_Randn` | `dtorch.randn` |
| （同上） | `_Arange` | `dtorch.arange` |
| （同上） | `_Full` | `dtorch.full` |
| （同上） | `_Randint` | `dtorch.randint` |
| （同上） | `_FromTorch` | `dtorch.from_torch` |
| `dtorch::core::DropoutOp` | `Dropout` | `dtorch.nn.functional.dropout` |
| `dtorch::core::EinsumOp` | `_Einsum` | `dtorch.einsum` |
| `dtorch::core::EmbeddingOp` | `Embedding` | `dtorch.nn.functional.embedding` |
| `dtorch::core::ExpandOp` | `_Expand` | `dtorch.nn.functional._expand` |
| `dtorch::core::FlattenOp` | `_Flatten` | `dtorch.flatten` |
| `dtorch::core::GetItemOp` | `_GetItem` | `dtorch.nn.functional._get_item` |
| `dtorch::core::InterpolateOp` | `Interpolate` | `dtorch.nn.functional.interpolate` |
| `dtorch::core::LinearOp` | `Linear` | `dtorch.nn.functional.linear` |
| `dtorch::core::MatmulOp` | `_Matmul` | `dtorch.matmul` |
| `dtorch::core::MaxMinOp` | `_Max` | `dtorch.max` |
| （同上） | `_Min` | `dtorch.min` |
| `dtorch::core::NormalizationOp` | `GroupNorm` | `dtorch.nn.functional.group_norm` |
| （同上） | `LayerNorm` | `dtorch.nn.functional.layer_norm` |
| （同上） | `RmsNorm` | `dtorch.nn.functional.rms_norm` |
| `dtorch::core::OuterOp` | `_Outer` | `dtorch.outer` |
| `dtorch::core::PadOp` | `Pad` | `dtorch.nn.functional.pad` |
| `dtorch::core::PermuteOp` | `_Permute` | `dtorch.permute` |
| `dtorch::core::PoolingOp` | `_Pooling2d` | `dtorch.nn.functional._pooling2d` |
| （同上） | `_GlobalPooling2d` | `dtorch.nn.functional._global_pooling2d` |
| `dtorch::core::ReduceOp` | `_Sum` | `dtorch.sum` |
| （同上） | `_Mean` | `dtorch.mean` |
| （同上） | `_Any` | `dtorch.any` |
| （同上） | `_All` | `dtorch.all` |
| `dtorch::core::RepeatOp` | `_Repeat` | `dtorch.nn.functional._repeat` |
| `dtorch::core::RepeatInterleaveOp` | `_RepeatInterleave` | `dtorch.repeat_interleave` |
| `dtorch::core::ReshapeOp` | `_Reshape` | `dtorch.reshape` |
| `dtorch::core::SdpaOp` | `_ScaledDotProductAttention` | `dtorch.nn.functional._scaled_dot_product_attention` |
| `dtorch::core::SetItemOp` | `_SetItem` | `dtorch.nn.functional._set_item` |
| `dtorch::core::SoftmaxOp` | `Softmax` | `dtorch.nn.functional.softmax` |
| `dtorch::core::SqueezeOp` | `_Squeeze` | `dtorch.squeeze` |
| `dtorch::core::TransposeOp` | `_Transpose` | `dtorch.transpose` |
| `dtorch::core::UnsqueezeOp` | `_Unsqueeze` | `dtorch.unsqueeze` |
| `dtorch::core::ViewOp` | `_View` | `dtorch.nn.functional._view` |
| `dtorch::core::WhereOp` | `_Where` | `dtorch.where` |
| `dtorch::core::MaskedOp` | `_MaskedFill`, `_MaskedScatter` | `dtorch.masked_fill`, `dtorch.Tensor.masked_fill/masked_scatter` |

### 2. 融合算子

定义在 `@dtorch/core/operators/fused_compile` 目录。

| C++ 核心算子 | C++ API 接口 | Python 接口 |
|---|---|---|
| `dtorch::core::ApplyRotaryEmbOp` | `_ApplyRotaryEmb` | `dtorch.nn.functional._apply_rotary_emb` |
| `dtorch::core::LayerNormMulAddOp` | `_LayerNormMulAdd` | `dtorch.nn.functional._layer_norm_mul_add` |
| `dtorch::core::SiluLinearChunkOp` | `_SiluLinearChunk` | `dtorch.nn.functional._silu_linear_chunk` |

### 3. 内部系统算子

定义在 `@dtorch/core/operators/system` 目录。

| C++ 核心算子 | C++ API 接口 | Python 接口 |
|---|---|---|
| `dtorch::core::MemoryOp` | `_EmptyCache` | `dtorch.nn.functional._empty_cache` |
| （同上） | `_GetMemoryStats` | `dtorch.nn.functional._get_memory_stats` |
| `dtorch::core::GetTensorOp` | `_GetTensorAsync` | `dtorch.Tensor.to_torch_async` |
| `dtorch::core::NvtxOp` | `_NvtxRangePush` | `dtorch.nn.functional._nvtx_range_push` |
| （同上） | `_NvtxRangePop` | `dtorch.nn.functional._nvtx_range_pop` |
| （同上） | `_NvtxMark` | `dtorch.nn.functional._nvtx_mark` |

### 4. Python 组合算子

以下算子不直接对应某个 C++ 核心算子，而是在 Python 层通过组合现有接口实现。

| Python 接口 | 实现说明 | 代码路径 |
|---|---|---|
| `dtorch.nn.functional.scaled_dot_product_attention` | 在 `_scaled_dot_product_attention` 基础上增加 context-parallel 支持 | `python/dtorch/nn/functional.py` |
| `_stack`、`_zeros_like`、`_ones_like`、`_full_like`、`_unbind`、`_argmax`、`_argmin`、`_nonzero` | 纯 Python 组合函数 | `python/dtorch/nn/functional.py` |

### 5. 补充说明

> - 表中 C++ API 接口省略了命名空间前缀 `dtorch::api::cpp::functional::`，所有接口均位于该命名空间下。
> - Python 接口列中，以 `dtorch.nn.functional._` 开头的接口未在 `__init__.py` 中重新导出为 `dtorch.*`，但仍可通过 `dtorch.nn.functional._xxx` 访问。
> - `dtorch.clip` 是 `dtorch.clamp` 的别名（定义在 `functional.py` 中：`_clip = _clamp`）。
