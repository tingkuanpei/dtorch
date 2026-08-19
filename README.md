# DTorch: Easy-to-Use Distributed Inference API for PyTorch

<p align="center">
| <a href="https://tingkuanpei.github.io/dtorch/en/"><b>Documentation</b></a> | <a href="https://tingkuanpei.github.io/dtorch/en/blog/introduction/"><b>Blog</b></a> | <a href="README.zh-CN.md"><b>中文</b></a> |
</p>

**DTorch is an easy-to-use distributed inference API for PyTorch.**, No multi-processes, no SPMD, no `ProcessGroup` setup — just write single-device code, then make one small change: replace `Tensor` by `DTensor` — DTorch will automatically handle resource management, scheduling, and communication.

---

## Example

For example, here is how to shard a tensor across two GPUs — the same task written both ways:

<table>
<thead>
<tr>
<th>DTorch (single-thread)</th>
<th>PyTorch (multi-process)</th>
</tr>
</thead>
<tbody>
<tr>
<td valign="top">

```python
# Launch: python3 test.py

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
# PyTorch needs torchrun to launch several processes.
# Launch: torchrun --standalone --nnodes=1 --nproc-per-node=2 test.py

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

## News

- [2026-08-17] Blog: [Introducing DTorch](https://tingkuanpei.github.io/dtorch/en/blog/introduction/) — an easy-to-use distributed inference API for PyTorch based on Single-Controller and Distributed Tensor.
- [2026-08-17] Blog: [DTorch Architecture: Combining Simplicity and Efficiency](https://tingkuanpei.github.io/dtorch/en/blog/architecture/)
- [2026-08-17] Blog: [DTorch: Advantages and Opportunities](https://tingkuanpei.github.io/dtorch/en/blog/advantages_and_opportunities/)

---

## Highlights

- 🔥🔥🔥 **Easy to Use.** By far the easiest-to-use distributed API for PyTorch — just like writing single-device PyTorch code. You describe the distributed computation as ordinary PyTorch code on a single thread, and the framework automatically handles task dispatch, scheduling, and communication across the cluster. (This makes it easy to orchestrate heterogeneous workloads — placing different models on different GPUs, or interleaving inference and training across RL rollouts and updates.)
- 🔥**Lower CPU overhead.** Benefiting from DTorch's asynchronous execution, the Python thread only constructs compute nodes — it never executes them directly. Combined with awaitable `TensorFuture`s, this keeps the interpreter light and minimizes GPU idle time caused by slow CPU-side kernel launches.
- **Single-Controller.** One python thread drives the entire cluster — no multi-process launch, no SPMD, no `ProcessGroup`.
- **DTensor Native.** Every tensor in DTorch is a `DTensor`, and every operator works on it natively. No manual shape sharding, no manual `all_gather` across ranks — you just declare the tensor and print it directly.
- **Unified parallel strategies.** Data Parallel, Tensor Parallel, Context Parallel (Ulysses & Ring) and Pipeline Parallel — all expressible in the same code, and **freely composable**. See the [Llama parallel guide](https://tingkuanpei.github.io/dtorch/en/user_guide/llama_parallel/) for a complete DP + TP + PP + CP example.

---

## Getting Started

DTorch currently only supports installing from source — see [GetStarted](https://tingkuanpei.github.io/dtorch/en/get_started/get_started/) for the build steps and examples.

Visit our [documentation](https://tingkuanpei.github.io/dtorch/en/) to learn more.

- [UserGuide](https://tingkuanpei.github.io/dtorch/en/user_guide/user_guide/) — understand DTensor core concepts and write distributed programs with the Python API.
- [DeveloperGuide](https://tingkuanpei.github.io/dtorch/en/developer_guide/project_overview/) — the architecture, design decisions, and internals of the DTorch engine.

---

## Call for Support

The [Single-Controller + Distributed Tensor architecture](https://tingkuanpei.github.io/dtorch/en/blog/architecture/) that DTorch builds on is a [promising technical direction](https://tingkuanpei.github.io/dtorch/en/blog/advantages_and_opportunities/). However, my personal resources alone are not enough to carry all of these ideas through. If you are interested in this direction, feel free to reach out at **peitingkuan@163.com**.

---

## Acknowledgment

DTorch's design and implementation draw on the ideas of [PyTorch](https://github.com/pytorch/pytorch), [OneFlow](https://arxiv.org/abs/2110.15032), [Pathways](https://arxiv.org/abs/2203.12533), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [vLLM](https://github.com/vllm-project/vllm), and [veRL](https://arxiv.org/html/2409.19256v1).
