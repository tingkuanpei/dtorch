# DTorch: Refactoring the PyTorch Distributed Ecosystem Through Ease of Use

DTorch is a distributed deep learning API built on the Single-Controller and Distributed Tensor architectures. The previous three blog posts introduced [the overall picture of DTorch](introduction.md), its [architecture design](architecture.md) and its [advantages and opportunities](advantages_and_opportunities.md). This article takes a different angle: instead of discussing DTorch itself, it discusses the ecosystem change DTorch may bring — **refactoring the PyTorch distributed ecosystem based on ease of use**.

## 1 Overview

The current PyTorch ecosystem is highly fragmented by task type: single-GPU training and inference use HuggingFace's [transformers](https://github.com/huggingface/transformers), [diffusers](https://github.com/huggingface/diffusers), [trl](https://github.com/huggingface/trl); multi-GPU inference uses [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang); multi-GPU training uses [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [DeepSpeed](https://github.com/microsoft/DeepSpeed); reinforcement learning uses [veRL](https://github.com/volcengine/verl) and [Slime](https://github.com/THUDM/slime). Every distributed framework maintains its own set of model implementations — the same model exists in the ecosystem as multiple mutually independent copies of code, each requiring its own maintenance; the optimizations of each framework are likewise bound to its own model implementations and cannot be reused across frameworks.

The proposition of this article is: **based on DTorch's ease of use, replace the technical foundation of the PyTorch distributed ecosystem — Multi-Controller + SPMD — with Single-Controller + DTensor, returning the whole distributed ecosystem to the shape of the single-device era** — only one model implementation, and distribution is merely a configuration of DeviceMesh and Placements.

The structure of this article is as follows: Section 2 analyzes the current state, causes, costs and essence of ecosystem fragmentation; Section 3 presents the refactoring plan (Single-Controller + DTensor replacing Multi-Controller + SPMD) and argues for it; Section 4 analyzes the opportunities for existing frameworks after the refactoring; Section 5 gives the implementation path and development cost analysis; Section 6 discusses the challenges; Section 7 concludes.

## 2 The current state of the PyTorch distributed ecosystem

### 2.1 Division of labor among frameworks

By task type, the current PyTorch ecosystem looks like Table 1:

<figure markdown>
  |Scenario|Frameworks|Approach|
  |-|-|-|
  |single-device training and inference|[transformers](https://github.com/huggingface/transformers), [diffusers](https://github.com/huggingface/diffusers), [trl](https://github.com/huggingface/trl)|provide an easy-to-use implementation of models and algorithms — the de facto "model standard library" of the ecosystem|
  |multi-GPU inference|[vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang)|each rewrites a set of parallel implementations of mainstream models, adding inference serving optimizations (PagedAttention, continuous batching, etc.)|
  |multi-GPU training|[Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [DeepSpeed](https://github.com/microsoft/DeepSpeed)|Megatron-LM provides dedicated parallel layers and requires model rewrites; DeepSpeed focuses on optimizations such as ZeRO, while TP / PP still need Megatron-style model adaptation|
  |reinforcement learning|[veRL](https://github.com/volcengine/verl), [Slime](https://github.com/THUDM/slime)|training engine and inference engine cooperate (introducing Ray to form a hybrid paradigm), handling weight synchronization between the two|

  <figcaption>Table 1: The PyTorch ecosystem fragmented by task type</figcaption>
</figure>

They solve different problems, but apart from single-device model implementations, **every distributed framework maintains its own set of model implementations**. This raises the threshold and time cost of adaptation: distributed code is hard to implement, adaptation must be completed by framework developers, and algorithm engineers can hardly do it themselves; cross-framework collaboration also requires writing large amounts of glue code to bridge the differences between distributed and single-GPU programs and across frameworks. Algorithm engineers ultimately cannot implement and iterate algorithms as quickly as on a single GPU.

### 2.2 The cause of fragmentation: evolution history driven by needs

Fragmentation was not designed by any organization — it grew out of needs. Reviewing the evolution of PyTorch's distributed interface (see [DTorch Introduction](introduction.md#322-evolution-history), Section 3.2.2 "Evolution history"), every new stage of demand gave birth to a new framework, as Table 2 shows:

<figure markdown>
  |Stage|Demand|Ecosystem response|
  |-|-|-|
  |Single-GPU era (2016~)|single-GPU training and inference|PyTorch eager + HuggingFace transformers / diffusers / trl, model implementations in one place|
  |DDP (2018)|Data Parallel|PyTorch DDP (SPMD paradigm), no model code changes needed|
  |TP / PP / EP (2019~2022)|large-model training|Megatron-LM appears; models need rewriting, weights need pre-sharding|
  |LLM inference (2023~)|high-throughput inference serving|vLLM and SGLang each rewrite mainstream model implementations and add inference optimizations|
  |Reinforcement learning (2024~)|RLHF / GRPO training|veRL and Slime appear; training and inference engines cooperate (hybrid paradigm with Ray)|

  <figcaption>Table 2: The evolution history of the PyTorch distributed ecosystem — every new stage of demand spawned a new framework</figcaption>
</figure>

Apart from the DDP stage, every new stage means "maintaining one more set of model implementations". The reason: PyTorch's distributed interface has been accumulating since DDP without ever having a unified design (see [DTorch Introduction](introduction.md#322-evolution-history), Section 3.2.2 "Evolution history") — only Data Parallel and ZeRO require no model code changes; all other forms of parallelism require customized modifications. So whenever new demands emerge, the community has no ready-made "distributed API" to use and can only create a new framework and re-implement the models.

### 2.3 The costs of fragmentation

The costs of fragmentation fall directly on every model and every user, as Table 3 shows:

<figure markdown>
  |Cost|Description|
  |-|-|
  |Multiple implementations of the same model|HuggingFace single-GPU version + Megatron training version + vLLM inference version; features and fixes need to be synchronized across the implementations|
  |Slow onboarding of new models|every framework adapts separately; the community often needs weeks to complete a model's distributed adaptation|
  |Weight format conversion|training and inference engines differ in sharding methods and parameter naming, requiring a checkpoint conversion toolchain|
  |Train-inference numerical mismatch|two sets of kernel implementations produce numerically different outputs from the same weights; costs the most in reinforcement learning|
  |Algorithm and engineering separated|distributed code can only be hacked by experienced engineers, and algorithm researchers use the hacked results ([Advantages and Opportunities](advantages_and_opportunities.md#3-easier-onboarding-of-new-algorithms), Section 3)|
  |High debugging cost|multi-process logs interleaved, breakpoints causing collective communication hangs, deadlocks hard to locate ([Advantages and Opportunities](advantages_and_opportunities.md#4-debugging-and-observability), Section 4)|

  <figcaption>Table 3: The costs of distributed ecosystem fragmentation</figcaption>
</figure>

### 2.4 The essence: a missing distributed API layer

Behind the surface of fragmentation lies a problem of layered structure. A deep learning ecosystem can be divided into three layers:

- **Operator library layer**: provides Tensors and operators (PyTorch operators, FlashInfer, FlashAttention, etc.);
- **Distributed API layer**: answers "how a Tensor is sharded and how computation is orchestrated across devices";
- **Solution layer**: vLLM, Megatron-LM, veRL, etc., solving problems in specific domains.

**The distributed API layer is missing.** What PyTorch currently provides for distribution is the SPMD approach — low-level primitives such as ProcessGroup and collective communication: only the mechanisms of processes and communication, with sharding, communication and scheduling all left to the user — essentially no solution at all. Apart from Data Parallel and ZeRO, all other forms of parallelism require customized modifications.

Consequently, every solution must rebuild this layer itself: vLLM developed its own parallel layers such as `QKVParallelLinear`, and Megatron-LM maintains `ColumnParallelLinear` / `RowParallelLinear` — **the same work has been repeated by every framework, and this duplicated set of parallel layers is in turn bound to each framework's own model implementations**. This is where the fragmentation comes from.

Conversely, once the distributed API layer is filled in, the solution layer no longer needs to rebuild it — this is exactly what DTorch sets out to do.

## 3 The refactoring plan

The current technical foundation of the PyTorch ecosystem's distributed layer is **Multi-Controller + SPMD**: one process per GPU, each process both controller and executor, code centered on rank, sharding and collective communication all completed manually. The missing API layer identified in [Section 2.4](#24-the-essence-a-missing-distributed-api-layer) is rooted in this foundation.

The refactoring plan: **replace the technical foundation of Multi-Controller + SPMD with Single-Controller + DTensor, leaving the model implementation layer untouched** — add DTensor support on top of HuggingFace's [transformers](https://github.com/huggingface/transformers), [diffusers](https://github.com/huggingface/diffusers), [trl](https://github.com/huggingface/trl), and the development experience returns to the single-device era.

### 3.1 Pillar one: Single-Controller, no more multi-process programs

Under Multi-Controller, the program launches N processes via torchrun, and the code must think in terms of rank throughout; the development and debugging costs of multi-process programs — interleaved logs, breakpoints causing collective communication hangs, deadlocks hard to locate — are all borne by the user ([Advantages and Opportunities](advantages_and_opportunities.md#4-debugging-and-observability), Section 4).

Under Single-Controller, the whole cluster has a single controller uniformly managing all GPUs: the user program is single-threaded and describes computation from a global view — no processes, no rank, no torchrun; sharding, communication and scheduling are all completed implicitly by the framework, and the code approaches a single-GPU program.

### 3.2 Pillar two: DTensor, no model code changes

Configuring the DeviceMesh and Placements of DTensor in a single-device program completes distributed support. For the usage of each parallel strategy at the Module layer, see [Module Parallel](../user_guide/module_parallel.md); for examples of freely combining and switching DP + TP + PP + CP, see [Llama Parallel Example](../user_guide/llama_parallel.md).

### 3.3 Extreme ease of use: back to the single-device era

The effect of the two pillars — Single-Controller + DTensor — combined is **extreme ease of use**:

- **Algorithm engineers** can freely implement model architectures as they wish and scale to multiple machines and GPUs for distributed training and inference — no longer needing to read, customize and debug obscure SPMD code;
- **Framework developers** are freed from the tedious development work imposed by SPMD and can invest their time in genuinely valuable work such as operator optimization and system optimization.

The overall development experience returns to the single-device era. Reviewing the history of deep learning frameworks, PyTorch's performance was often not superior, yet it became the de facto standard through ease of use. In the distributed era, this high ground of ease of use is still unoccupied — the same story is waiting to repeat itself.

## 4 Opportunities for existing frameworks: liberation, not replacement

After the switch to the Single-Controller + DTensor scheme, model implementations return to the standard implementations provided by HuggingFace, and existing distributed frameworks no longer need to maintain their own parallel model implementations. But the refactoring is not a replacement for them — it is empowerment: the distributed API layer filled in by DTorch takes over the heavy burden of distributed execution, and the way forward for each framework becomes clear:

**With DTorch's easy-to-use distributed API, frameworks are freed from the tedious multi-process scheduling and management of distributed execution and can invest their time in more valuable work such as operator-level / system-level optimization**. PagedAttention and the scheduling and fusion strategies in vLLM and SGLang, and the communication optimizations in Megatron, are core achievements honed by these frameworks over years; in the past they could only be delivered attached to a full set of model rewrites, but now they need only focus on the optimizations themselves, applied directly to HuggingFace's standard model implementations. **All optimizations become freely composable operators, no longer bound to any specific framework, and any model can use them**.

The competitive focus among frameworks shifts accordingly: from "who has the most complete model implementations" to "who has the deepest operator and system optimizations". For users, model implementations are no longer bound to any framework — **what users gain from the ecosystem refactoring is exactly this freedom of choice**: at training and inference time, they can pick the best combination of operators without migrating the whole model into a particular framework.

## 5 Implementation path and development cost analysis

### 5.1 Implementation path

The ecosystem refactoring proceeds from easy to hard with gradually increasing investment; every step has a clear landing point:

1. **Diffusion inference**: add DTensor support to the diffusers library, and diffusion models can run distributed inference on multiple machines and GPUs. DTorch has already migrated the SD3 / FLUX diffusion models based on `diffusers==0.34.0` as demonstrations;
2. **LLM inference**: add DTensor support to transformers — the changes are consistent with the diffusers demonstration — and complete the inference serving optimizations such as PagedAttention and continuous batching, reaching the performance level of vLLM;
3. **Complete support for training components**: beyond autograd and optimizers, parallel strategies such as DP / PP / EP / ZeRO all need dedicated support (such as overlapping computation and communication in training); once each is done, distributed training with transformers / trl becomes available;
4. **Complete support for RL training**: training and inference are two ways of using the same execution engine; multi-role orchestration is natively completed by Single-Controller, and the scenarios covered by veRL / Slime are then orchestrated in a unified way.

### 5.2 Development cost analysis

**Per-item staffing cost**: a team of 10 is expected to complete inference-related functionality, and a team of 15 to complete training-related functionality.

**Total cost comparison**: the refactoring is a one-time cost — the framework capability and the HuggingFace library integration each need to be invested only once; whereas the model adaptation cost under the status quo is paid repeatedly by every framework (N frameworks × M models, growing continuously as new models are released). One side is a one-time investment, the other a recurring investment that keeps inflating with the number of models — the gap in the ecosystem's total development cost is plain to see.

## 6 Challenges

For the refactoring plan to hold, the following challenges must be faced head-on:

- **DTorch's training and inference capabilities are still immature** ([Advantages and Opportunities](advantages_and_opportunities.md#10-disadvantages), Section 10): only the diffusion model inference demonstration has landed so far; the remaining steps in [Section 5.1](#51-implementation-path) — LLM inference, training components, RL training — are still roadmap;
- **DTorch only solves the distribution problem**: Single-Controller + DTensor solves "parallelism" and "ease of use", but operator optimization, request scheduling, and overlapping computation and communication are achievements accumulated by inference and training frameworks over years; they do not come automatically and still need to be integrated one by one after the refactoring;
- **Ecosystem inertia**: users and maintainers of existing frameworks face migration costs; the maturity and community accumulation of vLLM / Megatron will not evaporate overnight;

DTorch's positioning should be solutions, not merely a framework: proactively land solutions such as diffusion inference, LLM inference and RL training to promote the Single-Controller + DTensor paradigm, rather than passively waiting for other frameworks to integrate. Once the corresponding functionality is complete, DTorch can attract algorithm engineers with its ease of use and show its advantage in adapting new models.

## 7 Summary

The essence of the fragmentation of the PyTorch distributed ecosystem is a missing distributed API layer. DTorch implements an easy-to-use distributed API with the Single-Controller + DTensor scheme. Built on DTorch, distributed training and inference frameworks are freed from tedious multi-process scheduling and management, and the development experience and ease of use return to the single-device era.

Reviewing the history of deep learning frameworks, PyTorch's performance was often not superior, yet it became the de facto standard through ease of use. In the distributed era, this high ground of ease of use is still unoccupied — the same story is waiting to repeat itself.

## Further Reading

- [DTorch Introduction](introduction.md) — the comprehensive comparison of DTorch and PyTorch, the API introduction, precision and performance data
- [DTorch Architecture Design: How Simplicity and Efficiency Are Achieved Together](architecture.md) — the three core designs explained
- [Advantages and Opportunities of DTorch](advantages_and_opportunities.md) — advantages, industry opportunities, disadvantages and roadmap
