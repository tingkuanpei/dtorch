# Distributed Tensor (DTensor)

DTensor 的基本概念、常用 API 与 Module 级并行用法已在用户指南中介绍。本文从**开发者视角**补充三块进阶内容：**Operator 如何通过 PlacementSignature 推导输出分布**（及其"不自动注入 redistribute"的设计原则）、**不均匀切分**的内部处理，以及**加载 state_dict 时的自动切分与取值时的自动聚合**。

前置阅读：

- 基本概念（DeviceMesh、Placements、三种 Placement 策略、多维分布的读法）见 [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md)
- DTensor 的创建、算子推导、`redistribute` 等常用 API 见 [Python API Overview](../user_guide/python_api_overview.md)
- DP/TP/CP/PP 的 Module 级用法（内置并行层、Llama 完整示例）见 [Module 并行](../user_guide/module_parallel.md)

> PyTorch 中虽也支持 DTensor，但至今仍处于 [alpha 阶段](https://docs.pytorch.org/docs/stable/distributed.tensor.html)。

## 1. Operator 如何处理 DTensor

算子原生支持 DTensor、自动推导输出的 DeviceMesh/Placements、以及 Placements 不兼容时抛出异常的基本行为，见用户指南 [Python API Overview](../user_guide/python_api_overview.md)。本节深入其底层机制与设计原则。

**PlacementSignature —— 自动推导输出 Placement**

每个 Operator 内置一张 **PlacementSignature** 规则表，声明输入与输出 Placement 之间的映射关系。框架根据输入 DTensor 的实际 Placements 自动匹配规则，推导出输出 Placements（此过程**仅推导元信息，不触发任何实际通信**）。

**核心原则：不自动注入 redistribute**

> 当输入 Placements 无法匹配签名规则时，DTorch **默认抛出错误**，要求用户调整代码。

设计初衷：`tensor.redistribute()` 是开销很大的操作（底层涉及 all-gather、all-to-all、all-reduce 等集合通信）。若框架隐式执行，用户将失去对通信开销的感知与优化空间。因此用户需要**主动关心每个 Tensor 的 Placements 和 DeviceMesh，在必要时显式调用 `tensor.redistribute()`**。

**例外：少数算子自动注入通信**

为兼顾代码简洁性，少数算子会**自动插入 redistribute** 以简化常见场景。例如 `BroadcastBinaryOp`（加减乘除等二元运算）中，当 Replicate Tensor 与 Shard Tensor 运算时，框架会自动把 Replicate 转为 Shard（实现见 `dtorch/api/cpp/functional/implement/broadcast_op_imlp.cc` 中的 `PlacementR2S`）。这类自动注入通信的算子会在各自文档中着重说明。

> 更多细节见 [PlacementSignature](operator/placement_signature.md)。

## 2. 不均匀切分

DTorch 原生支持 Tensor 的不均匀切分——当某个维度的长度不能被设备数整除时，框架会**自动计算每个设备的本地 Shape**（而非报错或要求用户手动调整）。

```python
device_mesh = dtorch.DeviceMesh("cpu", [0, 1, 2, 3])

# 形状 [4, 11] 在第 1 维切分到 4 个设备：11 % 4 ≠ 0，为不均匀切分
x = dtorch.randn(4, 11, device_mesh=device_mesh, placements=[Shard(1)])
# GPU 0: 本地 shape [4, 3]
# GPU 1: 本地 shape [4, 3]
# GPU 2: 本地 shape [4, 3]
# GPU 3: 本地 shape [4, 2]  (少 1 列)
```

**redistribute 时的自动 padding**

NCCL 等集合通信库要求所有 rank 的输入 Tensor shape 完全一致。因此在执行 `tensor.redistribute()` 时，DTorch 会为不均匀切分的 Tensor **自动插入 padding 对齐，通信完成后再移除 padding**，整个过程对用户透明。例如将上述 `[Shard(1)]` 的 Tensor redistribute 到 `[Replicate()]` 时，框架会先补齐 GPU 3 的缺口，再执行 all-gather。

```python
# redistribute 自动处理 padding / unpadding
x_r = x.redistribute(device_mesh=device_mesh, placements=[Replicate()])
# 用户无需感知内部的 padding 逻辑
```

## 3. 自动切分与自动聚合

DTensor 的分布对用户透明：**加载权重自动切分，取值自动聚合**，二者互为逆操作。

**加载 state_dict —— 自动切分**

`state_dict` 中的权重是完整的 `torch.Tensor`，而目标 Parameter 已带有 Placements。框架按其 Placements 自动把完整权重切分到各 rank：`Shard(dim)` 沿对应维切分、`Replicate` 复制到每个 rank、`Partial` 仅 rank 0 保留真实值。用户加载一份完整的 state_dict 即可，无需像 Megatron-LM 那样按 rank 预先切分权重。

**取值 —— 自动聚合**

调用 `to_torch()` / `to_torch_async()` 取值时，框架自动把各 rank 上的分片聚合回一份完整 Tensor：`Shard(dim)` 沿对应维拼接、`Replicate` 取任一副本、`Partial` 做逐元素求和。用户无需手动调用 all-gather。
