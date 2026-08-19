# DTorch

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

## 特性亮点

- 🔥🔥🔥 **易用。** 目前最易用的 PyTorch 分布式 API —— 写起来就和单卡 PyTorch 代码一样。你在单线程中以普通 PyTorch 代码描述分布式计算，框架自动完成任务分发、调度与跨集群通信。（这也让编排异构负载变得轻松 —— 把不同模型放到不同 GPU 上，或在 RL 的 rollout 与训练更新之间交替执行推理。）
- 🔥**更低的 CPU 开销。** 得益于 DTorch 的异步执行，Python 线程只构建计算节点，从不直接执行它们。再配合可等待的 `TensorFuture`，解释器保持轻量，最大限度减少 CPU 侧 kernel 启动过慢造成的 GPU 空闲。
- **Single-Controller。** 一个 Python 线程驱动整个集群 —— 无需多进程启动、无需 SPMD、无需 `ProcessGroup`。
- **原生 DTensor。** DTorch 中的每个张量都是 `DTensor`，每个算子都原生作用于它。无需手动切分形状，无需跨 rank 手动 `all_gather` —— 声明张量后直接打印即可。
- **统一的并行策略。** Data Parallel、Tensor Parallel、Context Parallel（Ulysses 与 Ring）以及 Pipeline Parallel —— 都能用同一套代码表达，并且**可自由组合**。完整的 DP + TP + PP + CP 示例见 [Llama 并行指南](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/)。

---

## 文档

不知道从哪里读起？从这里开始：

- 🚀 **初次接触 DTorch？** 阅读[快速开始](get_started/get_started.md) —— 从源码构建 DTorch，然后验证 Llama 与扩散模型（SD3 / FLUX）的分布式推理。
- 📖 **想编写分布式程序？** 跟随 [User Guide](user_guide/user_guide.md)：DTensor 核心概念 → [Python API](user_guide/python_api_overview.md) → [Module 并行](user_guide/module_parallel.md) → 完整的 [Llama DP + TP + PP + CP 示例](user_guide/llama_parallel.md)。
- 🏗️ **想理解引擎？** 从[项目概述](developer_guide/project_overview.md)和[关键概念](developer_guide/key_concept.md)入手，再读[设计决策](developer_guide/design_decisions.md)与四层结构的 [Eager Graph 架构](developer_guide/eager_graph_architecture/eager_graph_architecture.md)。[文档索引](developer_guide/document_index.md)列出了其余全部文档。
- ⚡ **性能** —— 见[性能分析与优化](developer_guide/debug_alignment_optimization/performance.md)指南。

---

## 寻求支持

DTorch 所基于的 [Single-Controller + Distributed Tensor 架构](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/)是一条[颇具潜力的技术路线](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/)，然而仅凭个人的资源，还不足以将所有构想一一实现。如果你对这一方向感兴趣，欢迎联系 **peitingkuan@163.com**。
