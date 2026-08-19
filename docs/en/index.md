# DTorch

**DTorch is an easy-to-use distributed inference API for PyTorch.** No multi-processes, no SPMD, no `ProcessGroup` setup — just write single-device code, then make one small change: replace `Tensor` with `DTensor` — DTorch will automatically handle resource management, scheduling, and communication.

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

## Highlights

- 🔥🔥🔥 **Easy to Use.** By far the easiest-to-use distributed API for PyTorch — just like writing single-device PyTorch code. You describe the distributed computation as ordinary PyTorch code on a single thread, and the framework automatically handles task dispatch, scheduling, and communication across the cluster. (This makes it easy to orchestrate heterogeneous workloads — placing different models on different GPUs, or interleaving inference and training across RL rollouts and updates.)
- 🔥**Lower CPU overhead.** Benefiting from DTorch's asynchronous execution, the Python thread only constructs compute nodes — it never executes them directly. Combined with awaitable `TensorFuture`s, this keeps the interpreter light and minimizes GPU idle time caused by slow CPU-side kernel launches.
- **Single-Controller.** One Python thread drives the entire cluster — no multi-process launch, no SPMD, no `ProcessGroup`.
- **DTensor Native.** Every tensor in DTorch is a `DTensor`, and every operator works on it natively. No manual shape sharding, no manual `all_gather` across ranks — you just declare the tensor and print it directly.
- **Unified parallel strategies.** Data Parallel, Tensor Parallel, Context Parallel (Ulysses & Ring) and Pipeline Parallel — all expressible in the same code, and **freely composable**. See the [Llama parallel guide](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/) for a complete DP + TP + PP + CP example.

---

## Documentation

*The guides below are currently written in Chinese; English translations are on the way.*

Not sure where to start?

- 🚀 **New to DTorch?** Read [Get Started](https://tingkuanpei.github.io/dtorch/cn/get_started/get_started/) — build DTorch from source, then verify distributed inference of Llama and diffusion models (SD3 / FLUX).
- 📖 **Want to write distributed programs?** Follow the [User Guide](https://tingkuanpei.github.io/dtorch/cn/user_guide/user_guide/): DTensor core concepts → [Python API](https://tingkuanpei.github.io/dtorch/cn/user_guide/python_api_overview/) → [Module Parallel](https://tingkuanpei.github.io/dtorch/cn/user_guide/module_parallel/) → the complete [Llama DP + TP + PP + CP example](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/).
- 🏗️ **Want to understand the engine?** Start with the [Project Overview](https://tingkuanpei.github.io/dtorch/cn/developer_guide/project_overview/) and [Key Concepts](https://tingkuanpei.github.io/dtorch/cn/developer_guide/key_concept/), then read the [Design Decisions](https://tingkuanpei.github.io/dtorch/cn/developer_guide/design_decisions/) and the four-layer [Eager Graph Architecture](https://tingkuanpei.github.io/dtorch/cn/developer_guide/eager_graph_architecture/eager_graph_architecture/). The [Document Index](https://tingkuanpei.github.io/dtorch/cn/developer_guide/document_index/) lists all the remaining documents.
- ⚡ **Performance** — see the [Performance Analysis and Optimization](https://tingkuanpei.github.io/dtorch/cn/developer_guide/debug_alignment_optimization/performance/) guide.

---

## Call for Support

The [Single-Controller + Distributed Tensor architecture](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/) that DTorch builds on is a [promising technical direction](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/). However, my personal resources alone are not enough to carry all of these ideas through. If you are interested in this direction, feel free to reach out at **peitingkuan@163.com**.
