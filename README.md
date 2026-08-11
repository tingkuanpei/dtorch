# DTorch: An easy-to-use distributed inference API for PyTorch

<p align="center">
| <a href="https://tingkuanpei.github.io/dtorch/"><b>Documentation</b></a> | <a href="https://tingkuanpei.github.io/dtorch/"><b>Blog</b></a> |
</p>

**DTorch is an easy-to-use distributed inference API for PyTorch.** No multi-processes, no SPMD, no `ProcessGroup` setup — just write single-machine, single-GPU code, then make one small change: replace `Tensor` by `DTensor`. DTorch automatically handles resource management, scheduling, and communication.

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
# PyTorch needs torchrun to launch several processes; each runs the code below.
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

- **PyTorch-compatible API.** On a single device, DTorch behaves identically to PyTorch — tensors, operators, modules, and dtypes map one-to-one.
- **DTensor Native.** Describe N-dimensional parallelism with `DeviceMesh` + `Placements`. All operators infer the output's placement automatically.
- **Unified parallel strategies.** Data Parallel, Tensor Parallel, Context Parallel (Ulysses & Ring), Pipeline Parallel and Expert Parallel — all expressible in the same code, and **freely composable**.

---

## How It Works

DTorch is organized around three core design concepts.

### 1. Single-Client Single-Controller Multi-Worker

An asynchronous execution model with three cooperating roles:

```
┌──────────────┐      async messages      ┌─────────────────┐     kernel queue     ┌───────────┐
│  Client      │ ───────────────────────> │   Controller    │ ───────────────────> │  Worker   │
│  (Python)    │                          │  (C++ MainNode) │                      │ (C++ Thr) │
│              │ <── sync only on value ── │                 │ <─ sync only on val ─│           │
└──────────────┘                          └─────────────────┘                      └───────────┘
```

The **Client** (a single-threaded Python process) creates DTensor symbols and operators, serialized and sent asynchronously. The **Controller** (one `MainNode` per cluster, one `WorkerNode` per machine) builds the logical graph and manages all compute resources. Each **Worker** is a C++ thread bound to a CUDA stream that executes kernels in order. The pipeline is fully asynchronous and **synchronizes only when a tensor value is read**.

### 2. Distributed Tensor (DTensor)

`DeviceMesh` describes the GPU topology as an N-dimensional, named grid; `Placement` describes how a tensor is distributed along each mesh dimension. Three placements cover every mainstream parallel scheme:

| Placement | Meaning |
|---|---|
| `Shard(dim)` | Split the tensor along axis `dim` across devices |
| `Replicate()` | Full copy on every device |
| `Partial()` | Each device holds a partial result (reduced later) |

### 3. Eager Graph Architecture

A four-layer engine — *graph representation → kernel runtime → Eager Graph engine → collective communication* — that exposes a simple eager API while gaining graph-level optimizations. The Client emits incremental sub-graphs asynchronously; the Controller rewrites and executes them with a three-level concurrency model (between graphs, between operators, and within operators).

---