# Advantages and Opportunities of DTorch

DTorch is a PyTorch distributed API built on Single-Controller and DTensor

The previous two blog posts introduced [the overall picture of DTorch](introduction.md) and its [architecture design](architecture.md). Building on them, this article systematically discusses DTorch's advantages and disadvantages relative to existing distributed frameworks, as well as the industry opportunities and outlook it faces.

## 1 Advantages of DTensor

The essential complexity of distribution lies here: after a Tensor is sharded onto multiple devices, the information of "which device holds which piece of data" must be recorded somewhere. If it is not recorded in the program (PyTorch's ordinary Tensor), it must be expressed explicitly by user code — manually computing local shapes, branching by `rank`, calling all-gather / all-reduce at the right time. Such code is entangled with the algorithm logic: verbose, error-prone, and hard to modify.

DTensor records the distribution information in the Tensor itself: **DeviceMesh** describes "which devices it is distributed on", **Placements** describes "how it is sharded". Operators automatically complete the sharding and communication according to these two attributes, making distributed code as concise and easy to use as a single-device program: the Tensor always appears in the code in its complete logical form, and "which devices it is distributed on, how it is sharded" are merely its attributes. DTorch natively supports DTensor: all operators directly accept DTensor inputs and automatically infer the outputs' DeviceMesh and Placements. This brings a chain of simplifications:

- **Weights sharded automatically on load**: load one complete state_dict, and the framework shards it to each rank automatically according to the Placements — no need to pre-shard and convert weights by TP / PP rank as in Megatron-LM;
- **Values gathered automatically on retrieval**: `to_torch()` automatically gathers the shards on each rank back into a complete Tensor — no manual all-gather;
- **Uneven sharding handled automatically**: when a dimension's length is not divisible by the device count, the local shape is computed automatically, and communication automatically pads / unpads — transparent to the user throughout.

> PyTorch has also implemented DTensor, but it is still in [alpha state and under development](https://docs.pytorch.org/docs/stable/distributed.tensor.html)

**How frameworks without DTensor solve this problem**. Megatron-LM and vLLM each have their own answer — providing a set of dedicated parallel layers with the communication hard-coded into the layer implementations:

| Framework | Approach | Cost |
|---|---|---|
| PyTorch | does not solve it; users shard and communicate manually | distributed code is verbose and error-prone |
| Megatron-LM | provides dedicated parallel layers such as `ColumnParallelLinear` / `RowParallelLinear`, communication hard-coded in forward | models must be rewritten with Megatron operators; weights must be manually sharded by TP / PP rank, checkpoint conversion needed between training and inference |
| vLLM | likewise maintains a set of parallel layers (`QKVParallelLinear`, etc.) and rewrites almost all mainstream models for distributed inference | each model keeps a parallel implementation in addition to the HuggingFace one, doubling the onboarding cost of new models |
| DTorch | distribution information is an attribute of the Tensor, operators support it natively | model code is identical to single-GPU, only DeviceMesh and Placements need configuring |

## 2 Advantages of Single-Controller

**With DTensor, Multi-Controller + SPMD is no longer needed.** Multi-Controller + SPMD is not an ideal design chosen proactively, but a product of the era when DTensor was absent: since each process can only see local data, let each process manage itself, coordinating with rank and collective communication. Once DTensor brings back a global-view description of the data, the user program is naturally "global", and the more natural approach is to let the single Controller directly consume this global information — data sharding, task dispatch, communication coordination and resource management are all completed by the framework. This is exactly Single-Controller (as stated in [Design Decisions](../developer_guide/design_decisions.md): the Multi-Controller approach is designed around ordinary Tensors; after switching to DTensor, Single-Controller has better ease of use).

In terms of user experience, Single-Controller abstracts the entire distributed cluster into **one thread**: users touch only one ordinary Python thread — no torchrun, no multi-process, no rank — describing the computation from a global view. Multi-Controller exposes the multi-process model directly to users: process launching and lifetime, each process's individual state, and the rank-differentiated execution paths all need to be understood and managed process by process. This is the most intuitive source of Single-Controller's ease of use.

On top of that, the framework takes over the coordination work previously borne by users, with advantages on two levels.

**Automatic communication and scheduling**. The Controller holds the global computation graph: `redistribute()` is an ordinary operator, the framework can choose the most efficient implementation, insert the necessary synchronization nodes (CUDA Events), and achieve the overlap of computation and communication; communication deadlocks can be detected and avoided at graph-construction time. Users do not need to create and manage a ProcessGroup, nor care where all-reduce should be inserted.

**System capabilities brought by the global view** — under Multi-Controller's SPMD paradigm, these capabilities require a great deal of manual coordination or are simply unattainable:

- **Global device management and device virtualization**: a DeviceMesh is merely a description of a set of Device IDs; the physical devices are uniformly managed by the framework, so multiple virtual devices can be mapped to the same physical device — **single-device distributed simulation** is a concrete embodiment of device virtualization (environment variable `DTORCH_DTENSOR_IN_SAME_DEVICE=1`, simulating 8 GPUs by default, completing the development and debugging of distributed programs on a single GPU);
- **Runtime dynamic addition and removal of compute nodes**: the Controller uniformly manages all Workers, so new compute nodes can be added to the cluster or removed while the program is running (combined with the task migration above), achieving elastic scaling of the cluster — an SPMD process group is fixed at `torchrun` launch time, and runtime scaling requires reorganizing the process group and restarting;
- **Fault diagnosis**: the Controller holds the global state, so "which device, which operator went wrong" can be observed and located centrally, instead of being scattered in each process's own logs;
- **Faulty device eviction and automatic recovery**: the Controller knows which devices each task executes on, so it can migrate tasks on a faulty device to healthy devices and continue, instead of failing and restarting the whole job;
- **Auto Parallel** (automatic parallel strategy search) and **JIT compilation** (compilation optimizations based on global information).

> Current progress: DTorch has implemented process failure detection and graceful shutdown based on gRPC bidirectional heartbeats (detection in seconds, avoiding resource leaks and deadlocks), laying the foundation for the capabilities above; automatic recovery and the rest are natural extension directions of the Single-Controller architecture.

### 2.1 Cross-device computation orchestration: RL training and inference

Single-Controller also solves a class of problems SPMD struggles with — placing different roles on different devices and orchestrating their execution. Reinforcement learning training (PPO / GRPO) is the most typical scenario: actor generation (inference), reference model inference, reward model inference and parameter updates (training) each occupy a group of GPUs, executing alternately with complex dependencies. In the Multi-Controller ecosystem, vLLM and veRL introduced Ray for central scheduling, forming a hybrid paradigm of Single-Controller + Multi-Controller, and still need dedicated mechanisms for weight synchronization between the training engine and the inference engine.

DTorch is natively Single-Controller: placing training and inference on different GPUs just uses different DeviceMeshes — it is merely a difference in configuration, involving no change of code logic (dividing an independent inference group and training group, or letting both share the same group of GPUs, the rest of the program is completely identical). All roles are orchestrated by the same Controller; data transfer between roles (activations, weights) is an ordinary DTensor operation (`redistribute`), with no cross-engine format conversion or synchronization protocol needed:

```python
# illustration: orchestration of different roles in RL training (training capability is on the roadmap)
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=["dp", "tp"])

policy  = Policy(config, device_mesh=mesh)    # actor / rollout
ref     = Policy(config, device_mesh=mesh)    # reference model (frozen)
trainer = Trainer(policy, device_mesh=mesh)   # parameter updates

for batch in data:
    responses = policy.generate(batch.input_ids)   # inference
    rewards   = reward_model(responses)            # inference
    trainer.update(policy, responses, rewards)     # training: weights updated in place,
                                                   # no cross-engine synchronization
```

## 3 Easier onboarding of new algorithms

In the distributed ecosystem, the cost of landing a new algorithm is extremely high — every new algorithm must be adapted separately in each framework. DTorch compresses the onboarding cost back to "close to a single-GPU program":

| Type of new algorithm | Approach in the traditional ecosystem | Approach in DTorch |
|---|---|---|
| new model architecture | adapt separately to the HF single-GPU version, the Megatron training version and the vLLM inference version | implement once in the single-GPU style (or migrate from HF), configure DeviceMesh / Placements and it runs distributed |
| new parallel strategy | rewrite the communication logic in SPMD code | express as a new Placements combination or a parallel Module; implemented once, all models benefit |
| new optimization techniques (quantization, caching, fusion) | implemented separately in each framework | integrated at the operator / framework level, switched on via ExecuteConfig, model code unchanged |

Taking DTorch's already-landed diffusion model inference as an example:

- Migrating SD3 / FLUX from `diffusers==0.34.0` required only the necessary modifications (replacing the module base class and imports, wrapping the tokenizer output, etc.), with the directory structure identical to the original library and the model code almost the same as the single-GPU version;
- Sage Attention quantization is integrated inside the SDPA operator, switched on with one line of `QuantizeConfig`, benefiting all models;
- First Block Cache residual caching, RoPE / SiLU-Linear / LayerNorm operator fusion are all enabled through `ExecuteConfig`, without touching the model code;

The more fundamental change is in the way of collaboration. In current practice, distributed code is usually hacked by experienced engineers, and algorithm researchers use the hacked results for training and inference — this wall has severely limited algorithmic innovation. DTorch pulls the shape of distributed code back in line with single-GPU code: algorithm researchers can directly read, modify and debug distributed programs, and the path from a new idea to validation on a cluster is greatly shortened. For research teams with frequent algorithm iterations, this means higher experimental throughput.

## 4 Debugging and observability

Debugging is an underestimated major part of distributed development. In the SPMD ecosystem, the debugging experience of multi-process jobs is poor: logs are interleaved by rank, stack traces are scattered in each process; breakpoints need to be attached to a specific process, while collective communication hangs the whole job when one process is stopped and the rest are waiting; when NCCL communication deadlocks, all processes stop waiting, and it is hard to get useful stack traces. Locating "which rank, which communication, which shape mismatch" often requires repeated bisection and reruns.

From the user's view, a DTorch distributed program is just a single-threaded, global-view ordinary Python program, and the debugging experience returns to the single-GPU era:

- **Direct debugging**: `print` directly outputs the DTensor's global shape and placements; breakpoints and stepping on the Client thread can trace the execution of the entire distributed program;
- **Single-GPU reproduction**: combined with single-device distributed simulation (see Section 2), problems in multi-GPU environments can be stably reproduced and repeatedly debugged on a single GPU;
- **Centralized observation**: the Controller holds the global state; logs, heartbeats and error messages are concentrated in one place instead of scattered in each process;

Thus, the debugging cost of distributed programs returns from "dedicated troubleshooting by experienced engineers" to "algorithm researchers can do it themselves" — which is the other side of the collaboration change discussed in Section 3.

## 5 A framework unifying single-machine and distributed computation

In the current PyTorch ecosystem, "single-machine code" and "distributed code" are two independently maintained implementations: single-GPU training and inference use HuggingFace transformers / diffusers; distributed training uses the rewritten versions of Megatron-LM / DeepSpeed; distributed inference uses the rewritten versions of vLLM / SGLang. The same model has at least two or three implementations, and after a new model is released, the community often needs weeks to complete the distributed adaptation in each framework.

In DTorch, single-machine and distributed are **the same code executing on different DeviceMeshes**:

- When the DeviceMesh has only one device, a DTorch program is an ordinary single-GPU program, with performance basically identical to PyTorch's (SD3 single-GPU inference end-to-end time -2.08%, see the [introductory blog](introduction.md));
- The same Llama implementation supports arbitrary combinations of DP, TP, PP, CP, switched purely by DeviceMesh and Placements configuration (see [Llama Parallel Example](../user_guide/llama_parallel.md));
- Scaling from a single machine with multiple GPUs to multiple machines is likewise just a different DeviceMesh configuration;
- Combined with the single-device distributed simulation capability of Section 2, distributed programs with any parallel combination can be developed and debugged on one GPU (e.g., simulating 16 virtual GPUs to run DP × TP × PP × CP), and deployed to a real cluster after confirmation.

| Scaling from single GPU to multi GPU | PyTorch ecosystem | DTorch |
|---|---|---|
| Code | switch to another framework implementation (Megatron / vLLM …) | the same code |
| What changes | rewrite the model, shard weights, manage communication | modify DeviceMesh and Placements |
| Debugging | debug in a multi-process environment | debug on a single GPU first, then scale; distribution can be simulated on a single GPU |
| Deployment | decide the parallel scheme in advance | choose the parallelism by available resources, switch at runtime |

The significance of unifying single-machine and distributed: **development and debugging cost** (get the algorithm logic working on a single GPU first, then switch the DeviceMesh to scale, no rewriting), **deployment flexibility** (the same code choosing DP+TP or adding PP+CP by resource scale), **maintenance cost** (one implementation, fixes and new features benefiting both single-machine and distributed simultaneously).

## 6 A framework unifying training and inference

Training and inference use the same operators and model structures, yet engineering has split them into two ecosystems: training with Megatron-LM / DeepSpeed, inference with vLLM / SGLang. This split brings three costs:

1. **Multiple model implementations**: the same model has one parallel implementation in the training framework and one in the inference framework; features and fixes need bidirectional synchronization;
2. **Weight format conversion**: the training output and the inference engine's weight format (sharding method, parameter naming) differ, requiring a checkpoint conversion toolchain that is costly to maintain and error-prone;
3. **Train-inference numerical mismatch**: the two sides have different kernel implementations and parallel sharding methods, so the same weights produce numerically different outputs in the two engines.

Train-inference mismatch costs the most in reinforcement learning. In RLHF / GRPO training, the rollout is executed by the inference engine and the policy update by the training engine; the numerical difference between the two sides (train-inference mismatch) introduces extra algorithmic noise and affects training stability, and the community has invested heavily in train-inference consistency alignment; weight synchronization between generation and training (such as veRL's 3D-HybridEngine) also needs dedicated communication and re-sharding mechanisms.

DTorch has no architectural gap between training and inference: **the two are two ways of using the same execution engine** — the same set of operators (LibTorch backend), the same model code, the same DeviceMesh and Placements parallel description. In an RL workflow, rollout and update are orchestrated within the same framework (see Section 2): weights are ordinary DTensors, and transfer between roles is a `redistribute` operation — no format conversion, no cross-engine synchronization protocol; train-inference consistency is guaranteed by construction — both sides call the same implementation of the same operator, not two separately aligned implementations.

> Currently DTorch has completed the distributed inference prototype of diffusion models (SD3 / FLUX) on this architecture, with training capability on the roadmap. Training and inference being carried by the same framework is DTorch's long-term structural advantage over the "training framework + inference engine" combination.

## 7 Built on LibTorch: controllable development cost

An operator library covering mainstream models takes years of engineering accumulation. DTorch uses PyTorch's C++ operator library LibTorch as its computation backend; most Kernels directly call the LibTorch API to complete the computation, and a new operator only needs one layer of "shell" implementing the data structures and scheduling:

- **Operators need no rewriting**: the vast majority of Kernels directly call the LibTorch API for computation; a small team of a few people can complete the development and iteration of features;
- **Precision aligned naturally**: DTorch and PyTorch call the same operator library and run the same CUDA kernels; with consistent inputs and computation logic, the outputs can be identical bit by bit, saving the long precision-alignment work;
- **Ecosystem reused directly**: the operator coverage automatically expands as the LibTorch version upgrades, and HuggingFace model code only needs minor modifications to migrate;
- **Community operator libraries reused directly**: the community has many high-performance operator libraries based on PyTorch custom operators (such as [SageAttention](https://github.com/thu-ml/SageAttention), FlashAttention). DTorch's Python Kernel supports calling Python code inside C++ Kernels, so these libraries can be reused directly without porting their kernels.

## 8 Usage-based GPU cloud services

There are two business models for GPU cloud services today: **pay per token (request)** and **pay per hardware usage time**.

- **Pay per token (request)**: the platform hosts models and provides inference services (such as the various large-model APIs), billing users by requests or generated tokens. Under this model only fixed, widely used models can run; the usage is fixed, but the platform can optimize around it carefully, with extremely high GPU utilization.
- **Pay per hardware usage time**: the platform provides GPU machines; users run PyTorch computation directly in docker containers, billed by time. Users can do anything, but GPU utilization depends on the users' own usage, usually the lowest.

| | pay per token | pay per hardware time |
|---|---|---|
| Models that can run | fixed mainstream models hosted by the platform | any model, any computation |
| Optimization responsibility | the platform optimizes carefully | the user's own responsibility |
| GPU utilization | extremely high | depends on the user, usually the lowest |

The two models occupy opposite ends: the former sacrifices flexibility for utilization, the latter sacrifices utilization for flexibility. When users want to run their own models and algorithms while hoping for high utilization, neither option satisfies.

DTorch's architecture makes a finer-grained billing model possible. The Client, Controller and Workers are decoupled through asynchronous messages, and the three roles' responsibilities happen to form the division of labor between the user and the cloud platform: **the Client builds compute nodes, closely tied to the business logic**, running on the user side; **the Worker only executes computation, holding no business logic**, a stateless execution unit; **the Controller holds the global computation graph, knowing what resources the computation needs and when they can be released**.

Workers support dynamic addition and removal (see "Runtime dynamic addition and removal of compute nodes" in Section 2): request CPU / GPU computation resources from the cloud platform dynamically when needed; release them once the computation completes. In this model, **the user is responsible for business logic, and the cloud platform is responsible for scheduling storage, computation and network**, selling hardware resources to users dynamically, on demand, by time slices. Meanwhile, since the scheduling authority is centralized in the platform, the cloud platform can deeply optimize storage, computation and network, continuously raising resource utilization — **users gain freedom close to "renting hardware", and the platform reaches utilization close to "token hosting".**

This is also the direction of the "cluster operating system" envisioned by Pathways. DTorch's current resource management is still a prototype with static per-job allocation; this path is the long-term space opened by the Single-Controller architecture.

## 9 Industry opportunities

The popular distributed deep learning training and inference frameworks today are basically all built on PyTorch; the ecosystem seems to have converged on PyTorch, but some opportunities actually remain:

1. **PyTorch's distributed ease of use is insufficient, and its DTensor support is incomplete**. PyTorch's DTensor is still in [alpha state and under development](https://docs.pytorch.org/docs/stable/distributed.tensor.html), and has not been widely used by mainstream large-model training and inference frameworks. There is a vacancy in the layer of easy-to-use distributed APIs with native DTensor support.
2. **PyTorch is positioned as a framework, not a solution**. Solutions such as vLLM, SGLang and veRL grew on PyTorch and each achieved success, showing that value capture happens at the "solution" layer. Relying on PyTorch, solutions for various domains can be derived — the layer DTorch can occupy is precisely "the base of distributed solutions".
3. **PyTorch's support for domestic (Chinese) GPUs is poor**. New hardware needs a complete distributed software stack, and adapting PyTorch and its upper frameworks one by one takes very long cycles. DTorch is small and evolves fast; the device layer and communication layer (CUDA, NCCL) are independent adaptation modules, so it can quickly provide distributed capability for new hardware and become one of the options for domestic GPUs' distributed software stacks.

To compete with PyTorch through differentiation, DTorch is positioned as a distributed computation solution for specific domains — such as inference of Diffusion Transformer models (SD3 / FLUX landed), quantized training and inference (Sage Attention), reinforcement learning training, etc. — rather than a complete general-purpose framework. Implementing solutions and open-sourcing the code proves and promotes the DTorch distributed API, attracting other developers to join DTorch's development. By focusing on solutions to concrete problems, DTorch can move fast in small steps, gradually increasing staffing and generating corresponding business value.

## 10 Disadvantages

Every technical direction has its costs. Of DTorch's disadvantages, the first two stem from the architecture itself, and the last two from the project stage:

- **Single point and throughput ceiling of centralized scheduling** (structural): the entire cluster depends on one Controller; if it crashes, the job fails; the construction and dispatch of compute nodes also go through the same Python thread — at the scale of thousands of GPUs, the message throughput of centralized scheduling may saturate before the GPUs do. This is also one of the reasons SPMD still dominates ultra-large-scale training. DTorch has implemented failure detection and graceful shutdown of Worker processes; Controller fault tolerance and controller sharding (the direction of [Pathways](https://arxiv.org/abs/2203.12533)) are long-term topics.
- **Round-trip latency of data-dependent control flow** (structural): branches on data such as `if loss < threshold:` must send the value from the Worker back to the Client, constituting a cross-process round trip and a synchronization point; `TensorFuture` async retrieval can overlap with computation, but cannot eliminate it. The denser the control flow, the more visible the cost.
- **Operator coverage and ecosystem maturity** (staged): operators must be integrated one by one, and the coverage is far smaller than PyTorch's; optimizations such as CUDA Graph and torch.compile need framework-level adaptation one by one; the complexity is concentrated in the C++ engine, raising the contribution threshold.
- **Training capability not yet complete** (staged): only diffusion model (SD3 / FLUX) inference has landed; autograd, optimizers and other training capabilities are still on the roadmap — the train-inference unification of Section 6 is currently architectural potential, not a real capability.

These disadvantages delineate DTorch's current boundary: not targeting ultra-large clusters of thousands of GPUs in the short term, but establishing itself on medium-scale, domain-specific solutions (such as diffusion model inference), expanding step by step as the ecosystem matures.

## 11 Current progress and roadmap

Currently, DTorch has run through the complete pipeline of distributed inference of diffusion models on a single machine with multiple GPUs, and the communication and scheduling facilities for multi-machine clusters are also close to completion:

- **DTensor and the operator system**: DeviceMesh, Placements (Shard / Replicate / Partial) and automatic output inference are ready; automatic weight sharding, automatic value gathering and uneven sharding are transparent to users;
- **Parallel strategies**: DP, TP, PP, Ulysses / Ring CP have landed; the same Llama implementation supports arbitrary combinations (see [Llama Parallel Example](../user_guide/llama_parallel.md));
- **Applications and optimizations**: SD3 / FLUX (diffusers 0.34.0) distributed inference verified; Sage Attention quantization, First Block Cache and operator fusion enabled via ExecuteConfig; single-GPU end-to-end performance basically identical to PyTorch (SD3 time -2.08%);
- **System and quality**: two execution modes — single-machine multi-thread / cluster multi-process (ZMQ + gRPC heartbeats); single-device distributed simulation and `TensorFuture` async retrieval; the three-layer test system uses PyTorch as the baseline, with outputs identical bit by bit.

The way forward follows "inference first, training later; capability first, scale later". In the short term, DTorch is positioned as a distributed inference framework for diffusion models; in the medium term, once training capability is complete, it can become a training framework for Transformer model pre-training and post-training; in the long term, it can evolve toward a "cluster operating system".

| Stage | Goal | Key content |
|---|---|---|
| Near term | complete inference | expand operator and model coverage, integrate the community's excellent fused operator implementations, align inference performance with the level of mature frameworks |
| Medium term | complete training | autograd and optimizers, quantized training, land train-inference unification, then orchestrate RL training (PPO / GRPO) |
| Long term | expand scale | Controller fault tolerance and controller sharding, automatic recovery of faulty devices, dynamic scaling; explore Auto Parallel, JIT compilation and usage-based cloud services |

## 12 Outlook

**Ease of use is crucial for deep learning frameworks.** Reviewing the history of deep learning frameworks, compared with other contemporary frameworks, PyTorch was often at a disadvantage in performance, but it attracted a large number of developers with its ease of use and gradually became the de facto industry standard. So went the story of the single-GPU era; in the distributed era, the high ground of ease of use is still unoccupied. DTorch, based on the Single-Controller + DTensor technical route, is more usable than PyTorch in distributed scenarios — aiming at "distributed programs written the same way as single-GPU programs", building operator implementations on top of LibTorch, at a controllable development cost, taking this as the breakthrough point to restructure the current Transformer model training and inference workflows, and to support new training paradigms that emerge in the future.

## 13 Summary

This article discussed DTorch's differentiated positioning from five angles — technical advantages, business model, industry opportunities, disadvantages and the progress path:

| # | Advantage | Core point |
|---|---|---|
| 1 | DTensor | distribution information is an attribute of the Tensor, operators support it natively, distributed code is identical to single-GPU |
| 2 | Single-Controller | the global view brings automatic communication scheduling, fault handling, dynamic scaling and RL orchestration |
| 3 | onboarding of new algorithms | onboarding cost returns to "close to a single-GPU program"; optimization techniques land at the operator layer, all models benefit |
| 4 | debugging and observability | distributed programs can be debugged, reproduced and observed like single-GPU programs |
| 5 | single-machine and distributed unified | the same code executes on different DeviceMeshes |
| 6 | training and inference unified | two uses of the same execution engine, train-inference consistency guaranteed by construction |
| 7 | controllable development cost | reuses LibTorch and the community ecosystem; a small team of a few people can develop and iterate |
| 8 | usage-based cloud services | Client / Controller / Worker form the division of labor between user and platform, gaining freedom and utilization at the same time |
| 9 | industry opportunities | the distributed ease-of-use vacancy, the solution-layer positioning, domestic GPU adaptation |
| 11 | progress and path | the diffusion model inference pipeline has landed, advancing along "inference first then training, capability first then scale" |
| 12 | outlook | workload complexity keeps amplifying the mismatch with SPMD; the high ground of ease of use is still unoccupied |

These advantages share the same architectural choice: **Single-Controller + DTensor**. It returns distributed programs to the shape of single-GPU programs (1–6), compresses the development cost to what a small team can bear (7), and naturally delineates the division of labor between users and cloud platforms (8) — beneath the apparent convergence of the PyTorch ecosystem, these are exactly where DTorch's opportunities lie (9, 12). At the same time, the disadvantages discussed in Section 10 are likewise the cost of this architectural choice: the throughput ceiling of centralized scheduling must be faced head-on as the scale evolves, and the gap in ecosystem and training capability means that in the short term DTorch establishes itself on domain-specific solutions and expands outward step by step — the progress and path of Section 11 is precisely the concrete arrangement of this expansion.

DTorch's code and documentation are both open source on [GitHub](https://github.com/tingkuanpei/dtorch); attention and participation are welcome.

## Further Reading

- [DTorch Introduction](introduction.md) — the comprehensive comparison of DTorch and PyTorch, the API introduction, precision and performance data
- [DTorch Architecture Design](architecture.md) — the three core designs explained
- [Single-Controller Architecture](../developer_guide/single_controller.md) / [Distributed Tensor](../developer_guide/distributed_tensor.md) — the mechanisms from the developer's view
- [Design Decisions](../developer_guide/design_decisions.md) — why DTensor + Single-Controller, the LibTorch backend
