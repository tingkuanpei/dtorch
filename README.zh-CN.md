# DTorch：易用的 PyTorch 分布式推理 API

<p align="center">
| <a href="https://tingkuanpei.github.io/dtorch/"><b>文档</b></a> | <a href="https://tingkuanpei.github.io/dtorch/blog/introduction/"><b>博客</b></a> | <a href="README.md"><b>English</b></a> |
</p>

**DTorch 是一套易用的 PyTorch 分布式推理 API。** 无需多进程、无需 SPMD、无需配置 `ProcessGroup` —— 只需编写单卡代码，再做一处小小的改动：用 `DTensor` 替换 `Tensor` —— DTorch 会自动处理资源管理、调度与通信。

---

## 示例

例如，下面演示如何将一个张量切分到两个 GPU 上 —— 同一个任务用两种方式实现：

<table>
<thead>
<tr>
<th>DTorch（单线程）</th>
<th>PyTorch（多进程）</th>
</tr>
</thead>
<tbody>
<tr>
<td valign="top">

```python
# 启动:python3 test.py

import dtorch

shape = (4, 3)
mesh = dtorch.DeviceMesh("cpu", [0, 1])
placements = [dtorch.Shard(0)]
x = dtorch.rand(shape, device_mesh=mesh,
                placements=placements)

print(f"{x=}")
```

</td>
<td valign="top">

```python
# PyTorch 需要 torchrun 启动多个进程。
# 启动:torchrun --standalone --nnodes=1 --nproc-per-node=2 test.py

import torch
import torch.distributed as dist

dist.init_process_group('nccl')
world_size = dist.get_world_size()
rank = dist.get_rank()

torch.cuda.set_device(rank)

shape = (4, 3)
rows = shape[0] // world_size
x = torch.rand((rows, shape[1]), device='cuda')

print(f"{rank=}, x: {x.shape}, {x.device}")

all_x = [torch.zeros_like(x) for _ in range(world_size)]
dist.all_gather(all_x, x)
all_x = torch.concat(all_x, dim=0)

print(f"{rank=}, all_x: {all_x.shape=}, {all_x.device}")

dist.destroy_process_group()
```

</td>
</tr>
</tbody>
</table>

---

## 新闻

- [2026-08-17] 博客：[DTorch 介绍](https://tingkuanpei.github.io/dtorch/blog/introduction/) —— 基于 Single-Controller 与 Distributed Tensor 的易用 PyTorch 分布式推理 API。
- [2026-08-17] 博客：[DTorch 架构设计：简洁与高效何以兼得](https://tingkuanpei.github.io/dtorch/blog/architecture/)
- [2026-08-17] 博客：[DTorch 的优势与机遇](https://tingkuanpei.github.io/dtorch/blog/advantages_and_opportunities/)

---

## 特性亮点

- 🔥🔥🔥 **易用。** 目前最易用的 PyTorch 分布式 API —— 写起来就和单卡 PyTorch 代码一样。你在单线程中以普通 PyTorch 代码描述分布式计算，框架自动完成任务分发、调度与跨集群通信。（这也让编排异构负载变得轻松 —— 把不同模型放到不同 GPU 上，或在 RL 的 rollout 与训练更新之间交替执行推理。）
- 🔥**更低的 CPU 开销。** 得益于 DTorch 的异步执行，Python 线程只构建计算节点，从不直接执行它们。再配合可等待的 `TensorFuture`，解释器保持轻量，最大限度减少 CPU 侧 kernel 启动过慢造成的 GPU 空闲。
- **Single-Controller。** 一个 Python 线程驱动整个集群 —— 无需多进程启动、无需 SPMD、无需 `ProcessGroup`。
- **原生 DTensor。** DTorch 中的每个张量都是 `DTensor`，每个算子都原生作用于它。无需手动切分形状，无需跨 rank 手动 `all_gather` —— 声明张量后直接打印即可。
- **统一的并行策略。** Data Parallel、Tensor Parallel、Context Parallel（Ulysses 与 Ring）以及 Pipeline Parallel —— 都能用同一套代码表达，并且**可自由组合**。完整的 DP + TP + PP + CP 示例见 [Llama 并行指南](https://tingkuanpei.github.io/dtorch/user_guide/llama_parallel/)。

---

## 快速开始

DTorch 目前仅支持从源码安装 —— 构建步骤与示例见[快速开始](https://tingkuanpei.github.io/dtorch/get_started/get_started/)。

欢迎访问我们的[文档](https://tingkuanpei.github.io/dtorch)了解更多。

- [用户指南](https://tingkuanpei.github.io/dtorch/user_guide/user_guide/) —— 理解 DTensor 核心概念，使用 Python API 编写分布式程序。
- [开发者指南](https://tingkuanpei.github.io/dtorch/developer_guide/project_overview/) —— DTorch 引擎的架构、设计决策与内部实现。

---

## 寻求支持

DTorch 所基于的 [Single-Controller + Distributed Tensor 架构](https://tingkuanpei.github.io/dtorch/blog/architecture/)是一条[颇具潜力的技术路线](https://tingkuanpei.github.io/dtorch/blog/advantages_and_opportunities/)，然而仅凭个人的资源，还不足以将所有构想一一实现。如果你对这一方向感兴趣，欢迎联系 **peitingkuan@163.com**。

---

## 致谢

DTorch 的设计与实现参考了 [PyTorch](https://github.com/pytorch/pytorch)、[OneFlow](https://arxiv.org/abs/2110.15032)、[Pathways](https://arxiv.org/abs/2203.12533)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[vLLM](https://github.com/vllm-project/vllm) 和 [veRL](https://arxiv.org/html/2409.19256v1) 等项目。
