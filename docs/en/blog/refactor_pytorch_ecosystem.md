# DTorch: Refactoring the PyTorch Ecosystem, Toward a GPU Cluster Operating System

> **Reading note: this article uses concepts such as Distributed Tensor, Single-Controller and Multi-Controller. Readers unfamiliar with them may first refer to: [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md), [Single-Controller and Multi-Controller](../developer_guide/single_and_multi_controller.md). For a code comparison between DTorch and PyTorch distributed, see the [GitHub page](https://github.com/tingkuanpei/dtorch#example).**

## 1 Overview

PyTorch currently offers several paradigms for distributed computing, and training and inference frameworks choose different combinations of them, as [Table 2](#table-2) shows. **Among them, the Native Tensor (the Tensor on a single GPU) + Multi-Controller (SPMD) combination is the most widely used, yet it is cumbersome to use, has a high barrier to entry, and offers extremely poor ease of use.** The Multi-Controller (SPMD) paradigm requires Python code to drive the local GPU directly, and the scheduling process is a black box to the outside world. A cluster management system cannot observe the computation running inside containers, nor can it migrate GPU containers efficiently ([Section 6.6](#66-existing-container-migration-schemes)), which is why average GPU cluster utilization is generally low<sup><a href="#note-1">Note 1</a></sup> — once a GPU is allocated, it cannot be reclaimed and rescheduled no matter how low its utilization is; and if any single machine fails, the entire job restarts.

To solve these problems, DTorch abandons SPMD entirely and **builds a concise, easy-to-use and complete distributed API on Single-Controller + DTensor**: users no longer need to orchestrate processes and communication, and distributed programming returns to the single-device era, as the example code in [Table 1](#table-1) shows. **This API is also the standard interface of a GPU cluster operating system, as [Figure 1](#image-1) shows — through it the cluster becomes aware of the computation content and uniformly schedules storage, compute and network resources, forming an easy-to-use, low-cost, high-performance and highly reliable GPU cluster**. Global annual spending on AI/GPU server hardware has reached hundreds of billions of dollars, and the industry urgently needs a GPU cluster operating system that can observe the computation content and uniformly schedule cluster resources; whoever builds it first will hold the foundation of the next-generation ecosystem. Although refactoring the PyTorch ecosystem requires substantial investment, raising cluster utilization by even 1% saves far more than the entire investment the refactoring requires.

The article is organized as follows: Section 2 analyzes the status quo, causes, essence and costs of PyTorch distributed ecosystem fragmentation; Section 3 introduces DTorch's distributed API; Sections 4 and 5 argue for the two cornerstones of the DTorch API — Why DTensor and Why Single-Controller; Section 6 analyzes the predicament of GPU clusters and unfolds the vision of a GPU cluster operating system; Section 7 depicts the ecosystem after the refactoring and lays out the implementation path, disadvantages and challenges; Section 8 concludes.

<figure markdown id="image-1">
  ![DTorch API: the standard interface of a GPU cluster operating system](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@latest/blog/gpu_cluster_os_en.png)
  <figcaption>Figure 1: GPU cluster operating system overview<br> Users describe computation logic on the Python Client; the logic is serialized into an Operator message queue and sent to the cluster for asynchronous execution. The cluster manages all GPU Workers uniformly and can schedule them efficiently</figcaption>
</figure>

<figure markdown id="table-1">
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
<figcaption>Table 1: DTorch vs. PyTorch code comparison <br> Creating and printing a distributed Tensor sharded along dimension 0; a DTorch program is single-threaded and does not need to consider each rank's behavior</figcaption>
</figure>

## 2 The PyTorch distributed ecosystem

This section first presents the fragmented status quo of the PyTorch distributed ecosystem, then reviews its needs-driven evolution history, and finally looks past the surface to the essence — a missing distributed API layer.

### 2.1 The status quo: a fragmented ecosystem

PyTorch currently offers several paradigms for distributed computing: for tensor representation, Native Tensor (the Tensor on a single GPU) and DTensor (Distributed Tensor); for control paradigms, Multi-Controller (SPMD) and Single-Controller + Multi-Controller (SPMD). For the concepts involved, see: [Distributed Tensor](../user_guide/distributed_tensor_overview.md), [Single-Controller and Multi-Controller](../developer_guide/single_and_multi_controller.md). Training and inference frameworks choose different combinations of these paradigms, as [Table 2](#table-2) shows. Each combination falls short in maturity, completeness or ease of use — none achieves all three:

- Native Tensor requires manually deriving and orchestrating the Tensor's behavior on every rank, inserting communication operators where appropriate, and manually aggregating the values from all ranks when printing the Tensor.
- The SPMD paradigm requires all ranks to execute the same code, and when ranks behave differently the distinction can only be expressed with if-else branches in the execution path; yet in parallelism schemes such as TP, PP and EP and in RL training workflows, the behavior of different ranks differs enormously, which sharply degrades the ease of use of PyTorch's distributed code.
- The Native Tensor + Multi-Controller (SPMD) combination is the most widely used, yet it is cumbersome to use, has a high barrier to entry, and offers extremely poor ease of use.
- DTensor has been in development for years and is still [alpha state and under development](https://docs.pytorch.org/docs/stable/distributed.tensor.html).
- In the distributed setting, vLLM introduces Ray to manage cross-machine processes for scheduling inference requests, and veRL does so for multi-role heterogeneous orchestration in RL workflows; together they form a Single-Controller + Multi-Controller (SPMD) hybrid paradigm that further raises the barrier to entry.

Divided by task type, the picture is as [Table 3](#table-3) shows: the same model exists as multiple mutually independent implementations across frameworks, and optimizations are bound to each framework's implementations and cannot be reused across frameworks. Drilling down further to concrete parallelism schemes such as DP, TP, PP, CP, EP and ZeRO, the scope and depth of support vary widely across frameworks and paradigms.

In such an ecosystem, distributed code is hard to implement, and model adaptation depends on experienced framework developers; cross-framework collaboration also requires large amounts of glue code to bridge the differences between distributed and single-device code and across frameworks. Algorithm engineers are shut out of the distributed world and cannot implement and iterate on algorithms as quickly as they can on a single GPU.

<figure markdown id="table-2">
  |Tensor representation \ control paradigm|Multi-Controller (SPMD)|Single-Controller + Multi-Controller (SPMD)|
  |-|-|-|
  |Native Tensor|Megatron-LM, DeepSpeed, SGLang|vLLM, veRL, Slime, Monarch|
  |Distributed Tensor|TorchTitan, PyTorch FSDP2|  |

  <figcaption markdown="span">Table 2: Where each framework in the ecosystem lands on "tensor representation × control paradigm". For the concepts involved, see: [Distributed Tensor](../user_guide/distributed_tensor_overview.md), [Single-Controller and Multi-Controller](../developer_guide/single_and_multi_controller.md)</figcaption>
</figure>

<figure markdown id="table-3">
  |Scenario|Frameworks|Approach|
  |-|-|-|
  |single-device training and inference|transformers, diffusers, trl|provide an easy-to-use implementation of models and algorithms — the de facto "model standard library" of the ecosystem|
  |multi-GPU inference|vLLM, SGLang|each rewrites a set of parallel implementations of mainstream models, adding inference serving optimizations (PagedAttention, continuous batching, etc.)|
  |multi-GPU training|Megatron-LM, DeepSpeed|Megatron-LM provides dedicated parallel layers and requires model rewrites, and TP / PP still need Megatron-style model adaptation; DeepSpeed focuses on optimizations such as ZeRO|
  |reinforcement learning|veRL, Slime|training engine and inference engine work together (introducing Ray to form a hybrid paradigm); since the training and inference frameworks implement models differently, weight conversion and synchronization between the two are needed|

  <figcaption>Table 3: The PyTorch ecosystem divided by task type</figcaption>
</figure>

### 2.2 The cause of fragmentation: a needs-driven evolution

Fragmentation was not designed by any organization; it grew out of needs. Reviewing the evolution of PyTorch's distributed interface (see [DTorch Introduction](introduction.md#322-evolution-history), Section 3.2.2 "Evolution history"), every new stage of demand gave birth to a new framework, as [Table 4](#table-4) shows:

<figure markdown id="table-4">
  |Stage|Demand|Ecosystem response|
  |-|-|-|
  |single-device era (2016~)|single-device training and inference|PyTorch eager + HuggingFace transformers / diffusers / trl, model implementations in one place|
  |DDP (2018~)|Data Parallel|PyTorch DDP (SPMD paradigm), no model code changes needed|
  |TP / PP / EP (2019~)|large-model training|Megatron-LM implements parallelism schemes such as TP, PP and EP on the SPMD paradigm. Models need rewriting and weights need pre-sharding|
  |LLM inference (2023~)|high-throughput inference serving|vLLM and SGLang implement parallelism schemes such as TP, PP and EP on the SPMD paradigm, each rewriting mainstream model implementations and adding inference optimizations|
  |reinforcement learning (2024~)|RLHF / GRPO training|veRL and Slime introduce Ray for central scheduling, coordinating the training and inference engines (a hybrid paradigm) and handling weight synchronization between them|

  <figcaption>Table 4: The evolution history of the PyTorch distributed ecosystem</figcaption>
</figure>

PyTorch's distributed interface has been carried over from DDP to this day and has never had a unified design. Whenever a new demand emerges, the community has no ready-made distributed API to reuse and can only create a new framework and re-implement the models — the direct cause of ecosystem fragmentation.

### 2.3 The essence: a missing distributed API layer

The PyTorch distributed ecosystem is fragmented because it lacks a complete, easy-to-use and unified distributed API. A distributed deep learning framework's ecosystem can be divided into three layers:

- **Operator library layer**: provides high-performance operator implementations (PyTorch operators, FlashInfer, FlashAttention, etc.);
- **Distributed API layer**: answers how a Tensor is sharded and how computation is orchestrated across devices;
- **Solution layer**: vLLM, Megatron-LM, veRL, etc., solving training and inference problems in specific domains.

**PyTorch's distributed API layer has long been missing.** It provides only low-level functionality such as process creation and ProcessGroup collective communication under the SPMD paradigm, leaving all sharding, communication and scheduling of Tensors to the user. The DTensor and Single-Controller solutions that users urgently need have progressed slowly because they fundamentally conflict with the established SPMD paradigm: DTensor has been in development for years and is still [alpha state and under development](https://docs.pytorch.org/docs/stable/distributed.tensor.html); Single-Controller can only create and manage remote processes with the help of Ray or Monarch, existing as a hybrid form of Single-Controller + Multi-Controller (SPMD).

Every training and inference framework has to define and implement its own distributed API layer, and then build parallel computation support and operator and pipeline optimizations on top of it. These implementations are mutually incompatible and cannot be composed. The cost of fragmentation is paid again and again by every model and every user — with every release, every new model and every round of experiments, as [Table 5](#table-5) shows:

<figure markdown id="table-5">
  |Cost|Description|
  |-|-|
  |Multiple implementations of the same model|HuggingFace single-GPU version + Megatron training version + vLLM inference version; features and fixes need to be synchronized across the implementations|
  |Slow onboarding of new models|every framework adapts separately; the community often needs weeks to complete a model's distributed adaptation|
  |Weight format conversion|training and inference engines differ in sharding methods and parameter naming, requiring a checkpoint conversion toolchain|
  |Train-inference numerical mismatch|two sets of kernel implementations produce numerically different outputs from the same weights; costs the most in reinforcement learning|
  |Algorithm and engineering separated|distributed adaptation depends on senior framework developers; algorithm engineers cannot do it themselves, and new ideas must wait for adaptation to finish before they can be validated on the cluster|
  |High debugging cost|the Tensor's distribution across GPUs must be derived by hand; multi-process logs interleaved; no breakpoint debugging (breakpoints cause collective communication hangs, and deadlocks are hard to locate)|

  <figcaption>Table 5: The costs of distributed ecosystem fragmentation</figcaption>
</figure>

## 3 DTorch: an easy-to-use distributed API

To solve the problems of PyTorch's poor distributed API ease of use, low GPU cluster utilization and poor reliability, DTorch designs a concise, easy-to-use and unified distributed deep learning API based on Single-Controller and DTensor (Distributed Tensor). A comparison of DTorch and PyTorch example code is shown in [Table 1](#table-1). The API's core capabilities include:

- **Single-threaded programming**: **the user describes computation from a global view in a single thread; process orchestration, communication and synchronization are all handled by the framework, distributed programming returns to the single-device era, and the development and debugging experience matches that of a single-GPU program**;
- **Parallelism is just configuration**: the same model code runs unchanged on a single GPU or under DP, TP, PP, CP and their combinations, with the degree of parallelism and the sharding described by DeviceMesh and Placements, so switching parallel schemes requires only a configuration change;
- **Optimizations compose freely**: optimizations such as quantization, caching and kernel fusion are orthogonal to parallelism schemes and can be layered on demand on the same model implementation; in reinforcement learning, orchestrating the roles is just a difference in DeviceMesh configuration, weight transfer between roles is an ordinary redistribute operation, and training and inference share the same model implementation — the costs listed in [Table 5](#table-5), such as weight conversion, weight synchronization and train-inference numerical mismatch, disappear as a result;
- **Lower CPU overhead**: the Client, Controller and Workers execute asynchronously; the Python Client's execution time is about 64% lower than PyTorch's, CUDA Kernels are launched more promptly, and the model's end-to-end latency even drops slightly — measured SD3 data is in [Table 6](#table-6).

DTorch's documentation and previous blog posts have covered it systematically:

- **For an overview of DTorch, its API, and precision and performance data, see: [DTorch Introduction](introduction.md)**;
- For DTorch's architecture design — how the scheduling overhead of Single-Controller is solved by overlapping, how DTensor is supported, and how the Eager Graph architecture unifies eager mode and graph mode — see: [Architecture Design](architecture.md);
- For how to implement various parallelism schemes at the Module layer so that the same code supports both single-device and distributed modes, see: [Module Parallel](../user_guide/module_parallel.md);
- For a complete example of enabling various parallelism schemes on the Llama model, see: [Llama Parallel Example](../user_guide/llama_parallel.md).

Sections 4 and 5 explain why DTensor and Single-Controller were chosen to build the DTorch API. Section 6 explains how to build a GPU cluster operating system on the DTorch API, uniformly managing and scheduling the cluster's storage, compute and network resources.

## 4 Why DTensor

DTorch designs an easy-to-use distributed API based on Single-Controller and DTensor. This section explains why the DTensor approach was chosen.

DTensor adds DeviceMesh and Placements attributes to Native Tensor to describe how a Tensor is sharded and stored across GPUs. A Native Tensor carries no distribution information, so sharding, communication and aggregation can only be orchestrated manually by the user, producing large amounts of redundant code; once DTensor puts the distribution information inside the Tensor, all of this work is done automatically by the framework:

- **The Tensor carries its own global description**: with a Native Tensor, how it is distributed across GPUs and what the global shape is can only be inferred manually by the user from context; a DTensor's Shape is the global shape, its distribution is explicitly described by DeviceMesh and Placements, and printing it reveals both;
- **Automatic sharding when loading weights**: load a complete state_dict and the framework shards it across ranks according to Placements, with no need to pre-shard and convert weights per TP / PP rank as Megatron-LM requires;
- **Automatic aggregation on retrieval**: `to_torch()` automatically aggregates the shards from all ranks back into a complete Tensor, with no manual all-gather;
- **Communication as an operator**: `redistribute()` is an ordinary operator; users need not create or manage ProcessGroups, nor call collective communication such as all-reduce manually; underneath, the framework picks the most efficient implementation based on the computation graph and network topology, and overlaps computation with communication automatically;
- **Automatic handling of uneven sharding**: when a dimension length is not divisible by the number of devices, the local shape is computed automatically, and padding / unpadding during communication is handled automatically, all transparent to the user.

More importantly, DTensor expresses parallelism schemes such as DP, TP, PP, CP, EP and ZeRO uniformly as combinations of DeviceMesh and Placements — **with the same model code, configuring different DeviceMesh and Placements turns on any combination of parallelism schemes, reducing the parallel strategy from a model implementation to a piece of configuration** (for the usage of each scheme at the Module layer, see [Module Parallel](../user_guide/module_parallel.md); for examples of freely combining and switching DP + TP + PP + CP, see [Llama Parallel Example](../user_guide/llama_parallel.md)).

## 5 Why Single-Controller

DTorch designs an easy-to-use distributed API based on Single-Controller and DTensor. This section explains why the Single-Controller approach was chosen.

The trade-off between Single-Controller and Multi-Controller + SPMD is not a new question: TensorFlow v1 in 2015 was exactly Single-Controller — a single client built the computation graph and drove execution across the whole cluster — but limited by the implementations of the time, centralized scheduling left the industry with the impression that "centralized scheduling = slow". Around 2018, all-reduce and DDP matured and PyTorch went all in on Multi-Controller + SPMD. That choice was justified at the time:

- A Native Tensor carries only local data, and no process can see the whole picture, so the most natural approach is for each process to manage itself and coordinate with the others through rank and collective communication; SPMD also fits Data Parallel naturally;
- the Controller sits on the same machine as the GPU, so scheduling goes only over the PCIe bus, without the overhead of centralized scheduling.

But the era of large models broke both premises. First, SPMD requires all ranks to execute the same code, and when ranks behave differently the distinction can only be expressed with if-else branches in the execution path; yet in parallelism schemes such as TP, PP and EP and in RL training workflows, the behavior of different ranks differs enormously, which sharply degrades the ease of use of PyTorch's distributed code. DTensor carries a global data description and can express various parallelism schemes concisely, fitting naturally with Single-Controller's global view — DTensor + Single-Controller is the more natural combination for the large-model era. Second, Single-Controller's scheduling overhead has been amortized by DTorch's asynchronous pipeline, measuring on par with Multi-Controller in practice (see Section 5.1).

Both premises have now failed, making Single-Controller the more natural choice. The rest of this section unfolds step by step: Section 5.1 argues that Single-Controller's scheduling overhead has been amortized, Section 5.2 shows its advantages in ease of use, Section 5.3 presents the system capabilities brought by the global view, Section 5.4 introduces PyTorch's Single-Controller solution with Monarch as an example, and Section 5.5 summarizes.

### 5.1 Scheduling overhead is no longer a problem

Historically, the main reason the industry rejected Single-Controller was scheduling overhead: with Multi-Controller the Controller sits on the same machine as the GPU and scheduling goes only over the PCIe bus; Single-Controller schedules remote GPUs over the cross-machine network.

The cost of Single-Controller has been overstated — the root cause of TensorFlow v1's slowness is not centralization itself, but that scheduling could not run in parallel with computation: the scheduling time was added directly to the latency of every step. TensorFlow v1's execution was organized around `session.run`: each call from the Client blocked synchronously, waiting for all computation of that step to finish; the Master dispatched the partitioned subgraph to each Worker as a whole rather than dispatching operator by operator. Even so, steps remained serial — the next step's scheduling could not overlap with the current step's computation (except for IO stages such as input prefetching); and refining centralized scheduling to per-operator dispatch would make every operator's scheduling decision require a cross-machine round trip, which is even less feasible. In either form, scheduling and computation could only run serially.

DTorch solves Single-Controller's scheduling overhead by overlapping scheduling with computation. A deep learning program consists of a series of Tensors and Operators, and metadata such as the data types and shapes of the Operators' output Tensors can be determined before the computation finishes. The framework can therefore construct and schedule subsequent operators while earlier Kernels are still computing, so that graph construction, scheduling and Kernel computation overlap. DTorch implements this as a three-level asynchronous pipeline of Single-Client Single-Controller Multi-Worker; see [DTorch Architecture Design: How Simplicity and Efficiency Are Achieved Together](architecture.md).

DTorch's measured data supports this conclusion, as [Table 6](#table-6) shows:

<figure markdown id="table-6">
  |Aspect|PyTorch|DTorch|Notes|
  |-|-|-|-|
  |Small operator end to end (`Shape=(1,)` add operator)|5.93us|8.14us<span style="color: #ff9800; font-weight: bold;">(+37%)</span>|as the Shape grows, the gap approaches zero; latency for small operators still has room for optimization|
  |Large operator end to end (SDPA operator)|baseline|essentially identical|operator computation dominates|
  |SD3 Python Client execution time|0.575s|0.206s<span style="color: #4caf50; font-weight: bold;">(-64.17%)</span>|the CPU time saved by asynchronous execution is used to overlap the system's scheduling overhead|
  |SD3 single-GPU inference end-to-end time|1.683s|1.648s<span style="color: #4caf50; font-weight: bold;">(-2.08%)</span>|CUDA Kernels are launched more promptly and densely, so latency even drops slightly|
  |SD3 peak memory|18.131GB|17.663GB<span style="color: #4caf50; font-weight: bold;">(-2.58%)</span>|intermediate Tensors are released promptly under asynchronous execution|

  <figcaption>Table 6: Scheduling overhead and performance comparison between DTorch and PyTorch</figcaption>
</figure>

### 5.2 The easiest-to-use distributed API

**DTorch's distributed API offers a development and debugging experience identical to the single-device era.** Single-Controller abstracts the whole distributed cluster into one large computer: programming in a single thread, the user can call on all of the cluster's compute resources — no torchrun, no multi-process, no ProcessGroup, no need to distinguish each rank's execution path, describing computation from a global view. Debugging likewise returns to the single-GPU era: set breakpoints and print Tensor values directly; no more interleaved logs, multi-process race conditions or deadlocks caused by collective communication; distributed code can even be developed and debugged on a single GPU with the multi-GPU environment simulated.

Single-Controller also solves a class of problems that SPMD handles poorly — the orchestration of heterogeneous roles. Reinforcement learning training (PPO / GRPO) is the most typical scenario: actor generation, reference model and reward model inference, and parameter updates each occupy a group of GPUs, execute in alternation, and have complex dependencies. In the Multi-Controller ecosystem, veRL introduces Ray for central scheduling precisely for this purpose, forming a hybrid Single-Controller + Multi-Controller paradigm and specifically addressing weight synchronization between the training and inference engines. DTorch is Single-Controller natively: each role simply uses a different DeviceMesh — a difference at the configuration level only, not a code change; all roles are orchestrated by the same Controller, and data transfer between roles (activations, weights) is an ordinary DTensor operation (`redistribute`), with no cross-engine format conversion or synchronization protocol.

### 5.3 System capabilities unlocked by the global view

Single-Controller holds the global computation graph and global device state, making possible a series of capabilities that under the Multi-Controller SPMD paradigm require extensive manual coordination or cannot be achieved at all:

- **Automatic communication and scheduling**: `redistribute()` is an ordinary operator; the framework automatically chooses the most efficient communication implementation, inserts the necessary synchronization points (CUDA Events), and overlaps computation with communication; communication deadlocks are detected and avoided at graph construction time. Users need not create or manage ProcessGroups, nor worry about where all-reduce should be inserted.
- **Global device management and device virtualization**: a DeviceMesh is merely a description of a set of Device IDs, and physical devices are managed uniformly by the framework, so multiple virtual devices can be mapped onto the same physical device.
- **Faulty device eviction and automatic recovery**: the Controller catches exceptions raised by device faults and migrates tasks to healthy devices to continue, avoiding the failure and restart of the whole job.
- **Runtime dynamic addition and removal of compute nodes**: the Controller manages all Workers uniformly and can add new compute nodes to the cluster or remove them while the program runs, achieving elastic scaling of the cluster — an SPMD process group is fixed at torchrun launch time, and runtime scaling requires reorganizing the process group and restarting.
- **Auto Parallel and JIT compilation**: automatic parallel strategy search and compilation optimization based on global information.

### 5.4 Monarch: PyTorch's Single-Controller solution

The PyTorch team has also recognized the value of the Single-Controller approach and developed [Monarch](https://github.com/meta-pytorch/monarch), a project with native Single-Controller support. Monarch's [blog post](https://pytorch.org/blog/introducing-pytorch-monarch/) introduces its two core features:

> - Remote Actors with scalable messaging: Actors are organized into collections called meshes, and messages can be broadcast to all members.
> - Fault tolerance based on a supervision tree: Actors and processes form a tree, faults propagate up the tree, providing good default error-handling behavior and supporting fine-grained failure recovery.

Monarch's core features are essentially the same as Ray's: creating and managing remote processes in a more convenient way (Actors) with built-in error handling. In essence, Monarch is still the hybrid Single-Controller + Multi-Controller (SPMD) paradigm — its advantage is good compatibility, being able to reuse existing code in Megatron-LM and SGLang; its disadvantage is that it exposes multiple processes directly to users, so ease of use remains poor for parallelism schemes such as PP and EP where rank behavior differs greatly.

Monarch is aware of these problems and proposes [Distributed Tensors in Monarch](https://meta-pytorch.org/monarch/stable/generated/examples/distributed_tensors.html). But the Distributed Tensors here differ from those in [Section 4](#4-why-dtensor): without DeviceMesh and Placements attributes, they are best understood as a Native Tensor running on multiple devices at once. This proposal is incompatible with existing implementations in the PyTorch SPMD ecosystem and is expected to have high scheduling overhead, but it is nonetheless a positive exploration.

### 5.5 Summary

Single-Controller is the next direction for distributed systems: vLLM and veRL introduce Ray for central scheduling, and PyTorch proposes Monarch — the industry has proven this repeatedly through its actions. To stay compatible with existing distributed code, the PyTorch ecosystem keeps SPMD, forming the hybrid Single-Controller + Multi-Controller (SPMD) paradigm; DTorch, starting from ease of use, abandons SPMD entirely, remains compatible only with single-device PyTorch code, and builds the easiest-to-use distributed API.

Extreme ease of use gives DTorch room to survive, but not yet an absolute advantage over the PyTorch ecosystem. The next section discusses how DTorch's DTensor + Single-Controller API becomes the standard interface of a GPU cluster operating system, supporting an easy-to-use, low-cost, high-performance and highly reliable GPU cluster.

## 6 Toward a GPU cluster operating system

Today's GPU clusters use containers (Pods) as the smallest scheduling unit: users request a given number of GPU containers of a given model from a container orchestration system such as Kubernetes and run PyTorch programs inside them. To the orchestration system, the PyTorch program in a container is a black box — it can see only hardware metrics such as memory usage and GPU utilization, not what computation is running inside. Because GPU clusters cannot freeze and migrate GPU containers efficiently ([Section 6.6](#66-existing-container-migration-schemes)), they cannot schedule efficiently. This scheduling approach has the following defects:

- **Low resource utilization**: the orchestration system cannot see the computation running in a container and can only allocate GPUs as whole cards or whole containers; once allocated, no matter how low the GPU's actual utilization is, the cluster cannot reclaim and reschedule it.

- **No fault tolerance or elastic scheduling**: PyTorch's ProcessGroup is fixed at startup, so on failure tasks cannot be migrated to healthy nodes, nor can compute nodes be added or removed dynamically at runtime; scaling and failure recovery both require restarting the entire job.

- **The PyTorch API exposes too many hardware details**: the cluster hands full control of the hardware to the PyTorch program, which in turn exposes hardware details such as CUDA Graph, Stream and Event directly to upper-layer code, deeply binding framework and model implementations to the underlying hardware; when onboarding a new chip, upper-layer code must be re-adapted and precision-verified one piece at a time — "replaceable hardware, plug-and-play new chips" is out of the question.

In the current PyTorch ecosystem, GPU cluster management and scheduling rely heavily on manual orchestration, and onboarding new hardware requires users to modify programs, verify and adapt them. The industry urgently needs an operating system that can observe PyTorch computation and automatically schedule and manage the entire GPU cluster.

DTorch adopts a scheduling scheme in which the Client, Controller and Worker are decoupled, communicating with each other through message queues, as [Figure 2](#image-2) shows. This architecture exactly meets the needs of a GPU cluster operating system, as [Figure 1](#image-1) shows: the computation logic described by the user on the Python Client is serialized into a stream of Operator messages and sent to the Controller, which completes scheduling and hands it to the Workers on the GPU cluster for execution:

- **The Client runs on the user side and is the user interface of the GPU cluster operating system**; it builds compute nodes and carries the business logic;
- **The Controller and the Workers both run on the cluster side and are the scheduling and execution units**: the Controller holds all information the Client program requests, such as the hardware topology and the global computation subgraph; a Worker is responsible only for executing computation and is a stateless execution unit;
- the cluster operating system supports connections from multiple Clients, serving multiple users simultaneously. Each Client has its own Controller;
- the "Operator message stream" and the "computation results returned" carry little traffic, and communication can overlap with computation, so the network requirement is modest and an ordinary Ethernet environment is sufficient.

**Users access the GPU cluster through the DTorch API, and the cluster works closely with the Controller, giving it the ability to evict, freeze and migrate Workers, enabling elastic scheduling, fault tolerance and other capabilities, and realizing an easy-to-use, low-cost, high-performance and highly reliable GPU cluster.**

<figure markdown id="image-2">
  ![Single-Client Single-Controller Multi-Worker architecture](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/client_controller_worker_en.png)
  <figcaption>Figure 2: The Client, Controller and Workers execute concurrently and communicate asynchronously through queues</figcaption>
</figure>

### 6.1 Improving resource utilization

The average utilization of today's GPU clusters is generally below 50%<sup><a href="#note-1">Note 1</a></sup>. A GPU cluster operating system built on the DTorch API can evict, freeze and migrate Workers, improving resource utilization through better scheduling. Its scheduling policy is intuitive: classify tasks by real-time requirement and priority, guaranteeing high-priority real-time tasks first, and filling idle resources with low-priority, non-real-time tasks (a lower price encourages users to mark training and inference tasks as non-real-time, which are scheduled automatically when resources are idle).

<a id="note-1"></a>

!!! note "Note 1: measurement basis and data sources for 'the average GPU cluster utilization is generally below 50%'"
    - The measurement basis for average GPU cluster utilization is the time average of the GPU Utilization reported by nvidia-smi over a period (say a month).
    - The "50%" figure is an empirical value from personal conversations with peers; no authoritative statistics have been found so far.
    - [Vexxhost](https://vexxhost.com/blog/gpu-utilization-ai-infrastructure/) (a cloud provider) says most AI clusters' GPU utilization is only 30–50%, without stating the data source or measurement basis;
    - [DevZero](https://www.devzero.io/blog/why-your-gpu-cluster-is-idle) (a GPU development environment vendor) titles its article "Why Your GPU Cluster Is 80% Idle" — average cluster utilization of about 20% — and attributes it to development tasks that "hold GPUs without running them": researchers reserve GPUs for a week but use them only 10–15% of the time, and the article likewise does not state its measurement basis;
    - [Run:ai](https://mlops.community/blog/this-open-source-tool-measures-gpu-cluster-utilization-heres-why-that-matters#:~:text=However%2C%20%28based%20on%20experience%20with%20POCs%20and,fully%20utilizing%20their%20GPU%20and%20AI%20hardware.) reports about 14%, the only one of the three with a measured basis: over the course of a year Run:ai interviewed dozens of enterprises, whose researchers estimated their average cluster utilization at as high as 62%; yet Run:ai's own measurements of its customers' clusters (POCs and new deployments) averaged only about 14%.

    - The figure is also closely tied to the tasks a cluster runs, model sizes and the type of team. Interactive development in Jupyter Notebook, and training and inference of small models, all drag down average cluster utilization. A cluster dedicated to LLM training and carefully optimized can reach over 90% utilization. Teams with strong engineering capabilities achieve high utilization; small teams achieve low utilization.

There are many reasons for low average GPU cluster utilization, each requiring separate analysis and targeted optimization; the specific causes and the cluster operating system's countermeasures are shown in [Table 7](#table-7). These strategies are not deep technology in themselves, but under the existing container (Pod)-based scheduling scheme, the cluster can neither see nor modify user code, so the optimizations cannot be implemented.

<figure markdown id="table-7">
  |Cause|Details|Countermeasure|
  |-|-|-|
  |Reservations and "holding without running"|resources are reserved for each team by quota; some members over-request to grab resources but do not use them|unified resource pool + quota + oversubscription: migrate low-priority tasks when resources are scarce; migrate and merge low-utilization tasks to reclaim resources|
  |Development and debugging|GPUs are used intermittently during development and debugging, dragging down average utilization|share GPUs among users, migrating tasks automatically on memory overflow; simulate distribution on a single GPU so debugging does not occupy multiple GPUs|
  |Non-compute-intensive workloads|tasks such as inference, small models and embedding consume memory but not compute, so GPU utilization is inherently low|mixed deployment: schedule memory-heavy and compute-heavy tasks onto the same GPU so they complement each other|
  |Fluctuating inference traffic|online inference rises and falls with user traffic; utilization drops sharply in troughs such as nights and weekends|elastic scaling: reclaim GPUs during troughs and assign them to low-priority tasks such as offline training and inference|
  |Data and I/O bottlenecks|model loading, data loading, data preprocessing and checkpoint writing|preload frequently used models into GPU memory or host memory; framework-side data prefetching and multi-threaded preprocessing; asynchronous checkpoints; the OS collects statistics and suggests optimizations to users|
  |Whole-card exclusivity|the K8s device plugin defaults to hard-isolated exclusivity: one card serves only one task, and when the task underuses it, the idle compute and memory can neither be shared nor combined|fine-grained sharing + global scheduling: MIG / time slicing / operator-level scheduling let multiple tasks share one card|
  |Fragmentation|idle GPUs are scattered across nodes; a single node has too few for a multi-machine multi-GPU job to start|topology-aware scheduling + defragmentation: place tasks according to communication needs; migrate low-priority tasks to consolidate idle GPUs|
  |Environment initialization|image pulling, container startup, torchrun startup and NCCL initialization take minutes|persistent Worker pool: new jobs take over in seconds, without the cost of creating containers and processes|
  |Idle on failure|a single node failure restarts the whole job|heartbeat detection finds faults within seconds; tasks migrate to healthy nodes and only the lost part is recomputed|
  |Inadequate framework optimization|inefficient framework-layer implementations; calls to unoptimized operators|provide deeply optimized frameworks and operator implementations; the OS collects operator-level statistics, feeds them back to users and performs targeted optimization|
  |Model binding|user programs bind to a specific GPU model, leaving other models idle|unified scheduling: automatically match the GPU model to task requirements, weakening model binding|

  <figcaption>Table 7: Causes of low GPU utilization and the cluster operating system's countermeasures</figcaption>
</figure>

### 6.2 Fault tolerance and elastic scheduling

PyTorch SPMD's process group is fixed at startup: any rank failure restarts the whole job, and scaling likewise comes at the cost of a restart. The cluster operating system turns fault handling and elastic scheduling of resources into ordinary scheduling operations.

- **Fault awareness**: the GPU cluster operating system can observe the computation content and access the underlying hardware, so it detects hardware faults earlier than an orchestration system that sees only container metrics.
- **Fault recovery**: when a fault occurs, Worker state falls into two categories, recoverable and unrecoverable. In the former case, the task migrates to a healthy Worker and continues, and upper-layer programs are completely unaware of the fault; in the latter case, an exception is thrown to the user's Client, which on receiving it restores the model from the model file or the last checkpoint and continues the remaining computation.
- **Elastic scheduling**: through the scalable interface at the DTorch API layer, tasks can be dispatched dynamically to multiple Workers for parallel computation; the cluster operating system can evict, freeze and migrate tasks, and executes the remaining work elastically according to current load and queue.

### 6.3 Onboarding new chips: transparent to users

In the PyTorch ecosystem, using new hardware is difficult because hardware details and adaptation responsibility are pushed down to users layer by layer:

- **API layer**: the PyTorch API directly exposes a large amount of hardware detail such as CUDA Graph, Stream, the NCCL backend and CUDA environment variables; when chip vendors adapt PyTorch, they add even more private interfaces for performance, deeply binding upper-layer code to specific hardware.
- **Framework layer**: frameworks such as Megatron-LM and vLLM are developed and validated on NVIDIA GPUs and provide a series of high-performance operators targeting NVIDIA GPUs only; some teams further maintain their own modified versions on that basis. Using other hardware requires switching to a vendor-customized version, which differs substantially from both the original and the team's own modified version, and the work of verification, adaptation and precision alignment falls entirely on the user.
- **Deployment layer**: users launch programs as containers, and must adapt, install and verify item by item the hardware drivers, the software stack a chip needs to run and third-party dependencies. Hardware faults and troubleshooting are also the user's own responsibility.

DTorch, by contrast, fully hides hardware details behind a standard API, and the adaptation work for new hardware is taken over by the cluster operating system as a whole. When users want to switch to other hardware, they simply notify the cluster; upper-layer software needs no changes, and software stack adaptation, precision verification and the rest are all borne by the platform. **New chips are plug-and-play, transparent to users.** This model benefits users, cluster platforms and chip vendors alike:

- **Users**: switch seamlessly between hardware options, lowering the cost of use.
- **Cluster platforms**: can move user traffic as a whole to different hardware, reducing operational risk and cost and attracting more users.
- **Chip vendors**: gain a unified channel for bringing chips into production — a single adaptation to the cluster operating system, with no need to adapt to each training and inference framework one by one or to help users migrate tasks individually.

### 6.4 An excellent user experience

A GPU cluster operating system based on DTorch is easy to use, low-cost, high-performance and highly reliable:

- **Easy to use**: DTorch provides a single-device API consistent with PyTorch, so distributed programming is as simple as single-GPU programming. There is no need to buy expensive equipment or configure a cumbersome GPU environment: install the DTorch library and configure an API Key on any networked device — even a cheap Raspberry Pi — and the vast remote GPU compute is at your disposal.
- **Low cost**: billing is based on the user's real usage; unified scheduling amortizes the per-unit compute cost; users can switch seamlessly between hardware options and choose cheaper compute.
- **High performance**: the cluster schedules storage, network and compute uniformly, significantly improving GPU utilization; deeply optimized frameworks and operator implementations speed up user programs; the operating system can also collect runtime information about programs and suggest which optimizations users can apply.
- **High reliability**: efficient fault awareness and fault recovery provide users with highly available GPU services.

### 6.5 A new business model: from selling hardware to selling compute

In 2026, global AI/GPU server hardware spending has reached hundreds of billions of dollars, and GPU clusters are the most important infrastructure of the AGI era. For GPU cluster platforms, a cluster operating system built on the DTorch API means a new business model: **from selling hardware to selling compute**.

- **Selling hardware**: GPU machines are allocated by container (Pod), and programs running in the containers are a black box to the platform;
- **Selling compute**: compute resources are allocated according to the sequence of computations the user requests, and the final computation results are returned.

In the former, once a container is allocated, the platform must guarantee exclusive resources even when idle; in the latter, the platform retains the right to schedule resources and only needs to deliver correct computation results to users, so it can schedule freely across the whole cluster and improve resource utilization.

Under the "selling compute" model, users get an easy-to-use, low-cost, high-performance and highly reliable service; the platform sells compute services, earns a reasonable profit, and has ample incentive to reduce operating costs: more efficient scheduling, hardware with lower adaptation cost, and better training and inference frameworks and operator implementations. Chip vendors gain a unified channel to production — one adaptation suffices, with no need to adapt to each training and inference framework one by one. Thus a business model emerges in which users, platforms and chip vendors all win.

### 6.6 Existing container migration schemes

Existing task migration schemes fall into two categories: container destruction and recreation, and container live migration. The former is the current mainstream approach, at the cost of interrupting the task; the latter is far from mature — neither gives the cluster an efficient way to migrate containers.

**1. Container destruction and recreation**

The cluster orchestration system asks the user program to save a checkpoint (assuming the user has implemented that logic), then destroys the container and restarts the task on a new machine. The process cannot be transparent to users and has clear defects:

1. the user must implement the checkpoint-saving logic;
2. saving the checkpoint and restarting the task is time-consuming (pulling images, setting up the runtime environment, loading model files — typically about ten minutes), during which the machine sits idle;
3. if the user has not implemented checkpoint saving, the migration also loses part of the computation.

In practice, cluster platforms rarely destroy user containers on their own — doing so forcibly interrupts user tasks and creates trouble for the platform itself; a platform usually only sends a warning email when it notices low GPU utilization, and most users ignore it.

**2. Container live migration**

Live migration is still at the "early commercial / industrial pilot" stage; the most mature approach is [CRIU](https://github.com/checkpoint-restore/criu) + [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint). It has the following defects:

- **Slow migration**: it must migrate GPU memory, CUDA process state, and container memory and state, and the time grows linearly with GPU memory usage — per [DevZero's measurements](https://www.devzero.io/blog/gpu-container-checkpoint-restore), generating the cuda checkpoint alone takes about 13 seconds on a single A100 and about 55 seconds on 4 cards; the checkpoint file is comparable in size to the process's GPU memory footprint, and cross-machine transfer and restoration take additional time;
- **Migration is possible only between environments with identical software and hardware**: any difference in CPU or GPU model, driver version, CUDA version, etc. causes the migration to fail — the official CUDA documentation requires the GPU on the restoring side to be the same chip model as the original with sufficient memory ([CUDA Checkpointing Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html)), and combining NCCL with CUDA Checkpoint carries strict library-version constraints ([NCCL Roadmap](https://github.com/NVIDIA/nccl/issues/2272));
- **NCCL is not yet supported**: NCCL, which multi-machine distributed training depends on, is out of scope ([DevZero](https://www.devzero.io/blog/gpu-container-checkpoint-restore#current-limitations-and-requirements)); NCCL and gloo communication groups must be rebuilt by adapting the user program, and multi-machine migration requires additional coordination — NVIDIA itself lists "improved CUDA Checkpoint support (removing deviceAPI, CUDA Graph and strict library-version constraints)" as a not-yet-started work item ([NCCL Roadmap](https://github.com/NVIDIA/nccl/issues/2272)).

**3. DTorch's Worker migration**

Compared with the two schemes above, Worker migration in a GPU cluster operating system is very efficient. A Worker is a stateless execution unit, so migration only needs to copy the Tensors required for the computation from one GPU to another, and it can be performed between any hardware; the Controller automatically rebuilds the NCCL and gloo communication groups, and user programs are completely unaware of it.

- **Inference**: model weights are frozen and can be loaded from the model file in advance, so only Activation Tensors need to be copied;
- **Training**: weights and Adam optimizer states, among others, must be copied — the copy can be timed after the weight update completes, avoiding copying gradients and Activation Tensors; when fp32 master weights and fp16 copies coexist, only the fp32 weights need copying, since the fp16 copies can be re-derived from them by conversion;
- memory/GPU memory reserved in PyTorch's memory pools and CPU memory need not be copied;

Taken together, **container migration does not solve the problem of low GPU utilization — it merely moves the problem from one machine to another**: it solves none of the problems of low GPU utilization listed in [Table 7](#table-7). More critically, task migration needs to be paired with "observing the computation inside containers" and be able to migrate across GPUs of different models and merge multiple tasks onto a single GPU; only then can fragmented GPU memory and compute be consolidated, improving overall cluster utilization. A task's compute and GPU memory demands fluctuate over time; if the cluster cannot observe the computation inside containers, it can only base merging decisions on utilization — when compute demand rebounds, tasks contend with each other and latency increases, and once memory demand exceeds the capacity of a single GPU, the tasks fail with OOM errors.

## 7 Refactoring the PyTorch ecosystem through ease of use

DTorch builds an easy-to-use distributed API on DTensor and Single-Controller. With its extreme ease of use, distributed training and inference frameworks built on the DTorch API can attract a share of users. This API is also the standard interface of a GPU cluster operating system — through the easy-to-use, low-cost, high-performance and highly reliable compute service the cluster provides, DTorch can attract most users.

### 7.1 The opportunity

Today's mainstream distributed training and inference frameworks are almost all built on PyTorch; the ecosystem appears to have converged on PyTorch, but opportunities remain:

- **Insufficient distributed ease of use**: PyTorch's distributed capabilities are built on Multi-Controller (SPMD); parallelism schemes such as TP / PP / EP require rewriting models, manually sharding weights and orchestrating communication, and distributed development depends on senior framework engineers; the officially promoted DTensor has been in development for years and is still alpha state and under development, not widely adopted by mainstream training and inference frameworks.
- **Low cluster utilization and poor reliability**: a PyTorch program in a container is a black box to the scheduling system; GPUs can only be allocated as whole cards, and idle compute cannot be reclaimed and rescheduled; the process group is fixed at startup, so any rank failure restarts the whole job ([Section 6.1](#61-improving-resource-utilization), [Section 6.2](#62-fault-tolerance-and-elastic-scheduling)).
- **High cost of onboarding new hardware**: the PyTorch API exposes hardware details such as CUDA Graph and Stream, and frameworks such as Megatron-LM and vLLM are deeply bound to NVIDIA GPUs; new hardware requires building a complete distributed software stack and adapting to PyTorch and each upper-layer framework one by one, a long cycle with duplicated investment across vendors ([Section 6.3](#63-onboarding-new-chips-transparent-to-users)).

**DTorch has two core advantages: extreme ease of use, and the ability to improve GPU cluster utilization.** As [Section 6.5](#65-a-new-business-model-from-selling-hardware-to-selling-compute) notes, global annual AI/GPU server hardware spending has reached hundreds of billions of dollars. Raising cluster utilization by even 1% saves far more than the entire investment required to refactor the ecosystem. Extreme ease of use is the entry point that attracts users to DTorch; the cost advantage from higher cluster utilization is DTorch's trump card.

**PyTorch's distributed API is still incomplete and its distributed ecosystem is cumbersome to use, which leaves DTorch a rare window of opportunity.** PyTorch itself is a compute API rather than a solution: solution frameworks such as transformers, diffusers, trl, Megatron-LM, vLLM, SGLang and veRL all grow on top of it; what users really need are those frameworks that run models directly, and the compute API is merely the foundation that carries them. Building frameworks with the same functionality on the DTorch API can attract a large number of users through extreme ease of use — PyTorch itself defeated TensorFlow on ease of use back then, and now the balance of ease of use will tip toward DTorch. New models will keep emerging in different domains, and once DTorch and PyTorch stand at the same starting point, ease of use will become the decisive factor.

GPUs are expensive: a server with 8 H800 cards already costs 2 million RMB, and most universities, laboratories and small companies cannot afford to build their own GPU clusters; even when renting, the vast majority of teams lack a senior AI Infra team to put expensive GPUs to efficient use. A GPU cluster operating system based on DTorch offers another option: no need to own GPUs, just use them on demand — a single API summons a cluster that previously only top teams possessed, and all the tedious GPU-related work is borne by the cluster. **More importantly, users pay only for the computation actually executed, not for idle compute.** Expensive GPUs and senior AI Infra optimization are no longer the preserve of top teams but a shared resource available to everyone — efficient compute within reach will greatly unleash the innovative potential of researchers.

### 7.2 Implementation path

The ecosystem refactoring proceeds from easy to hard in three steps: inference, training and the GPU cluster operating system. A few people can start it, then gradually build influence and increase investment.

**Inference**

- Diffusion inference (demonstration completed): add DTensor support to diffusers, matching the functionality of SGLangDiffusion. DTorch has already migrated SD3 / FLUX based on `diffusers==0.34.0` as a demonstration;
- LLM inference: add DTensor support to transformers (the changes are consistent with the diffusers demonstration), and complete inference serving optimizations such as PagedAttention and continuous batching to reach the performance level of vLLM.

**Training**

- Training components: autograd, optimizers and parallelism schemes such as DP / PP / EP / ZeRO need to be supported (such as overlapping computation and communication in training); once done, distributed training with transformers / trl becomes available;
- RL training: training and inference are two ways of using the same execution engine, and multi-role orchestration is natively completed by Single-Controller.

**GPU cluster operating system**

Core functionality:

- Resource pooling: the cluster's compute, memory, network and storage form a unified resource pool, allocated, reclaimed and shared at task and operator granularity;
- Global scheduler: topology-aware placement, priority and preemption, load balancing, task eviction, freezing and migration, dynamic scaling;
- Hardware abstraction layer: hides differences of GPU models and vendors, so new chips need to adapt only to this layer;
- Fault management: heartbeat detection finds faults within seconds, tasks migrate to healthy nodes and only the lost part is recomputed;
- Multi-tenant isolation: isolation of compute, memory and data between tasks, with quota and permission management;
- Observability and billing: collect operator-level runtime statistics, show users where optimizations are possible, and bill by actual usage.

### 7.3 Why not PyTorch

PyTorch can hardly replicate DTorch's two core advantages:

**1. Ease of use of the distributed interface**

PyTorch's distributed interface is centered on the Multi-Controller + SPMD paradigm, which inherently conflicts with ease of use. [Section 5.2](#52-the-easiest-to-use-distributed-api) has already argued why an easy-to-use distributed interface must be built on Single-Controller, so it is not repeated here.

**2. Scheduling for a GPU cluster operating system**

- PyTorch's Python code and CUDA execution must live in the same process; running CUDA kernels on a remote machine is not supported, and adding this capability would be very costly to retrofit. Moreover, separating scheduling from execution introduces higher latency, which requires a systematic solution.
- The SPMD paradigm requires multiple processes, one per GPU: distributed training on 128 GPUs, for example, needs 128 processes. Run them on the user side and the local machine is overwhelmed; run them on the cluster side and starting and debugging jobs becomes inconvenient for the user. In addition, 128 processes must communicate with the cluster and execute in sync, incurring enormous communication overhead and latency.

PyTorch originally chose SPMD precisely because scheduling stayed on the local machine: the Controller sat on the same machine as the GPU, so scheduling went only over the PCIe bus, with no cross-network overhead. But when the Python program and the CUDA kernels live on different machines, scheduling is bound to cross the network — the premise on which SPMD rested has vanished, so what reason remains to keep it?

### 7.4 A new ecosystem where everyone wins

Once the refactoring is complete, one easy-to-use distributed API will run through the whole technology stack: upward it supports training and inference frameworks, downward it serves as the standard interface of the GPU cluster operating system, so that the cluster offers an easy-to-use, low-cost, high-performance and highly reliable compute service. The division of labor in the stack becomes clear again, and everyone in the ecosystem benefits:

**1. Users**

- Users call on the GPU cluster through a distributed API as easy to use as single-device, with no need to orchestrate processes and communication; development and debugging return to the single-GPU era. Distributed programming no longer depends on senior framework engineers, and algorithm engineers can focus on algorithm and model optimization.
- GPUs are reachable from any networked device, with no image environment to configure and no containers to launch; the experience is no different from local computing. There is no need to build a GPU cluster: models and data are hosted in the cloud. Users pay only for the computation actually executed, not for idle compute. The GPU cluster optimizes automatically to speed up programs, and offers optimization suggestions to help users improve efficiency.
- Users can switch hardware at any time: programs and frameworks need no changes, and software stack adaptation and precision verification are borne entirely by the platform.

**2. Solution developers**

- Building training and inference frameworks on the DTorch API means focusing only on pipeline and operator optimization — distributed details such as Tensor sharding, communication and scheduling are all handled by the API layer.
- Optimizations are decoupled from model implementations, freely composable and reusable across frameworks; in RL training, training and inference share the same model implementation, weight synchronization needs no format conversion, and training and inference are numerically consistent by nature.

**3. GPU cluster platforms**

- The platform offers an easy-to-use, low-cost, high-performance and highly reliable compute service, and its business model shifts from "selling hardware" to "selling compute" — no longer renting out exclusive GPUs by container, but allocating resources per computation request and delivering computation results.
- The right to schedule GPUs stays with the platform: evicting, freezing and migrating tasks and recovering from faults all become ordinary scheduling operations; the cluster schedules storage, compute and network resources uniformly, so the same hardware carries more computation and utilization keeps rising. Since the cluster can see the computation content, the platform can also perform deep optimization at the framework and operator layers to speed up user programs.
- The platform controls the definition of the API, and upper-layer frameworks and user programs access compute through it. The platform is therefore not bound to any chip vendor and can freely choose and adapt the chips with the best price-performance — better service at lower cost.

**4. Chip vendors**

- Chips gain a unified channel to production — adapt only to the cluster operating system's hardware abstraction layer, with no need to adapt to PyTorch and each upper-layer framework one by one or to help users migrate tasks individually.
- New chips are plug-and-play, and competition returns to the price-performance of the chips themselves.

### 7.5 Disadvantages and challenges

The refactoring plan also faces the following challenges:

- **DTorch's training and inference capabilities are still immature**: only the diffusion model inference demonstration has landed so far; the remaining work of the inference and training steps in [Section 7.2](#72-implementation-path) — LLM inference, training components and RL training — is still on the roadmap, and matching the performance of mature training and inference frameworks requires substantial development effort;
- **Ecosystem inertia**: users and maintainers of existing frameworks face migration costs; the maturity and community accumulation of frameworks such as Megatron-LM, vLLM, SGLang and veRL will not evaporate overnight;
- **The GPU cluster operating system is yet to be built**: the scheduling, storage, communication, fault tolerance, multi-tenancy and other functionality listed in [Section 7.2](#72-implementation-path) must all be built from scratch, requiring sustained investment and a certain development cycle;

DTorch first attracts a share of users with the ease of use of its distributed interface; once the GPU cluster operating system is built, it attracts most users with an easy-to-use, low-cost, high-performance and highly reliable compute service. The distributed API is the standard interface of the cluster operating system, and the cluster operating system provides the compute foundation for the API; the two complement each other and ultimately form a complete ecosystem.

## 8 Summary

DTorch's two core advantages — extreme ease of use and the ability to improve GPU cluster utilization — spring from the same source: the user describes the computation logic, and the framework holds a global description of the compute nodes. With that global description, all the tedious work is borne by the framework. Users no longer need to orchestrate processes and communication, and distributed programming returns to the single-device era. The cluster's scheduling system can see the computation content, so evicting, freezing and migrating tasks and recovering from faults all become ordinary scheduling operations, and new hardware needs to adapt only to a unified abstraction layer.

Global annual AI/GPU server hardware spending has reached hundreds of billions of dollars, and the industry urgently needs a GPU cluster operating system that can observe the computation content and uniformly schedule cluster resources; whoever builds it first will hold the foundation of the next-generation ecosystem. Ten years ago, PyTorch took the ecosystem from TensorFlow through ease of use; in the distributed era, the opportunity will belong to the easy-to-use, low-cost, high-performance and highly reliable GPU cluster operating system.

## Further Reading

- [GitHub page](https://github.com/tingkuanpei/dtorch) — the project source repository
- [DTorch Introduction](introduction.md) — a comprehensive comparison of DTorch and PyTorch, the API introduction, and precision and performance data
- [DTorch Architecture Design: How Simplicity and Efficiency Are Achieved Together](architecture.md) — the three core designs explained
- [Advantages and Opportunities of DTorch](advantages_and_opportunities.md) — advantages, industry opportunities, disadvantages and roadmap
