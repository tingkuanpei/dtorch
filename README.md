# DTorch: An easy-to-use distributed inference API for PyTorch

<p align="center">
| <a href="https://tingkuanpei.github.io/dtorch/"><b>Documentation</b></a> | <a href="https://tingkuanpei.github.io/dtorch/"><b>Blog</b></a> |
</p>

**DTorch is an easy-to-use distributed inference API for PyTorch.** No multi-processes, no SPMD, no `ProcessGroup` setup — just write single-device code, then make one small change: replace `Tensor` by `DTensor`. DTorch will automatically handle resource management, scheduling, and communication.

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

- 🔥 **Easy to Use.** By far the easiest-to-use distributed API for PyTorch — just like writing single-device PyTorch code. You describe the distributed computation as ordinary PyTorch code on a single thread, and the framework automatically handles task dispatch, scheduling, and communication across the cluster. (This makes it easy to orchestrate heterogeneous workloads — placing different models on different GPUs, or interleaving inference and training across RL rollouts and updates.)
- **Single-Controller.** One python thread drives the entire cluster — no multi-process launch, no SPMD, no `ProcessGroup`. 
- **DTensor Native.** Every tensor in DTorch is a `DTensor`, and every operator works on it natively. No manual shape sharding, no manual `all_gather` across ranks — you just declare the tensor and print it directly. 
- **PyTorch-compatible API.** DTorch provides an API consistent with PyTorch — tensors, operators, modules, and dtypes map one-to-one.
- **Unified parallel strategies.** Data Parallel, Tensor Parallel, Context Parallel (Ulysses & Ring), Pipeline Parallel and Expert Parallel — all expressible in the same code, and **freely composable**.
- **Asynchronous Computation.** DTorch uses asynchronous execution to overlap the distributed system's scheduling overhead with computation. This also provides a range of performance-optimization opportunities.
- **Lower CPU overhead.** Benefiting from DTorch's asynchronous execution, the Python thread only constructs compute nodes — it never executes them directly. Combined with awaitable `TensorFuture`s, this keeps the interpreter light and minimizes GPU idle time caused by slow CPU-side kernel launches.

---

## Architecture

DTorch rests on three core design concepts that work together to make distributed execution feel like ordinary single-device PyTorch. A **Single-Controller** execution model lets one Python thread drive the entire cluster. The **Distributed Tensor (DTensor)** data model expresses N-dimensional parallelism through `DeviceMesh` and `Placements` that every operator understands natively. And the **Eager Graph Architecture** is the asynchronous engine that turns these declarative descriptions into efficient, overlapping execution across devices. Under the hood, DTorch runs on **LibTorch** (PyTorch's C++ library) as its compute backend, matching PyTorch's operators, numerics, and performance.

### 1. Single-Client Single-Controller Multi-Worker

An asynchronous execution model with three cooperating roles:

```
┌──────────┐       async messages       ┌────────────┐       kernel queue        ┌────────┐
│ Client   │ ─────────────────────────> │ Controller │ ──────────────────────-─> │ Worker │
│ (Python) │                            │ (C++)      │                           │ (C++)  │
│          │ <── sync when get value ── │            │ <─ sync when get value -─ │        │
└──────────┘                            └────────────┘                           └────────┘
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