# Single-Controller and Multi-Controller

When scaling a deep learning program from a single GPU to multiple GPUs and machines, the system must answer a fundamental question: **who decides which GPU executes which part of the computation?** By where the control authority is placed, mainstream distributed systems fall into two paradigms: **Multi-Controller** (one controller per GPU, used by PyTorch) and **Single-Controller** (one controller for the whole cluster, used by DTorch). This article introduces these two concepts as background knowledge for the rest of the developer guide.

## 1. What is a Controller

A deep learning program consists of a series of Tensors and Operators; when running on a single GPU, one process directly drives the GPU. After scaling to multiple GPUs, some role is needed to:

- Shard (or replicate) Tensors across devices
- Decide which Operators each device executes
- Insert all-reduce, all-gather and other collective communications at the right time
- Manage resources such as devices, processes, and communication groups (ProcessGroups)

The role that takes on these responsibilities is the **Controller**. Different numbers and placements of controllers yield the two paradigms:

<figure markdown>
  ![Single-Controller vs Multi-Controller](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/single_controller_vs_multi_controller.png)
  <figcaption>Figure 1: Single-Controller (left) and Multi-Controller (right)</figcaption>
</figure>


## 2. Multi-Controller

**Multi-Controller** starts one process per GPU; each process is both controller and executor: it directly drives the local GPU and coordinates with other processes through collective communication (NCCL). All processes run the **same code**, each processing different data — the **SPMD** (Single Program Multi Data) paradigm. Representative systems: PyTorch `torch.distributed` (DDP / FSDP), Megatron-LM, DeepSpeed.

Taking PyTorch as an example, sharding two Tensors onto 4 GPUs and adding them:

```python
# Launch: torchrun --nproc_per_node=4 main.py
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")        # create the process group manually
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
rank, world_size = dist.get_rank(), dist.get_world_size()

x = torch.randn(256, 128, device="cuda")
y = torch.randn(256, 128, device="cuda")
x = x.chunk(world_size)[rank]                  # manual sharding: this rank takes only its own data slice
y = y.chunk(world_size)[rank]

z = x + y                                      # each process computes its own slice independently (local view)
```

Characteristics of Multi-Controller:

- **Local view**: the code thinks in terms of `rank` everywhere — "which process am I? which slice of the data is mine?"
- **Manual coordination**: process group creation, data sharding, and collective communication calls are all done by the user
- **Very low scheduling overhead**: each process controls its local GPU only through the PCIe bus

## 3. Single-Controller

**Single-Controller** keeps only one controller in the whole cluster, which uniformly manages all GPU resources. The user describes the computation from a **global view** in one ordinary process, and data sharding, task dispatch and communication coordination are all done automatically by the framework. This model was first used in TensorFlow v1 and is also discussed in [Pathways](https://arxiv.org/abs/2203.12533); representative systems: TensorFlow v1, Pathways, and DTorch.

The same "shard onto 4 GPUs and add" described with DTorch:

```python
import dtorch
from dtorch.distributed_spec import init_device_mesh, Shard

mesh = init_device_mesh("cuda", (4,), mesh_dim_names=["dp"])           # declare the cluster topology

x = dtorch.randn(256, 128, device_mesh=mesh, placements=[Shard(0)])    # the framework shards automatically
y = dtorch.randn(256, 128, device_mesh=mesh, placements=[Shard(0)])

z = x + y                             # global view: one line describes the computation of the whole cluster
```

No `torchrun`, no `rank` branches, no manual sharding — the Tensor always appears in code in its complete logical form; "which devices it is distributed on" is merely its Placements attribute.

Characteristics of Single-Controller:

- **Global view**: consistent with the user's way of thinking, distributed code is close to a single-GPU program
- **Automatic coordination**: sharding, communication, and resource management are done implicitly by the framework
- **Holds the global state**: the controller holds the global computation graph and can perform global optimizations such as computation-communication overlap and operator fusion

The cost is higher scheduling overhead: controlling GPUs on remote machines goes through the inter-machine network. DTorch amortizes this overhead through asynchronous execution.

## 4. Further Reading

For a detailed comparison of the two paradigms on usability, flexibility, scheduling overhead and other dimensions, see [Single-Controller Architecture](single_controller.md). Other reading paths:

- [Key Concepts](key_concept.md) — an overview of DTorch's three core designs
- [Design Decisions](design_decisions.md) — why DTensor + Single-Controller
- [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md) — an introduction to the DeviceMesh and Placements concepts
