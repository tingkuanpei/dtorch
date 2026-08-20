# DTorch：基于易用性，重构 PyTorch 分布式生态

DTorch 是一套基于 Single-Controller 和 Distributed Tensor 架构的分布式深度学习 API。前三篇博客分别介绍了 [DTorch 的整体情况](https://tingkuanpei.github.io/dtorch/cn/blog/introduction/)、[架构设计](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/) 与 [优势与机遇](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/)。本文换一个视角：不再讨论 DTorch 本身，而是讨论它可能带来的生态变化——**基于易用性，重构 PyTorch 分布式生态**。

## 1 概述

当前的 PyTorch 生态按任务类型高度碎片化：单卡训练和推理用 HuggingFace 的 [transformers](https://github.com/huggingface/transformers)、[diffusers](https://github.com/huggingface/diffusers)、[trl](https://github.com/huggingface/trl)，多卡推理用 [vLLM](https://github.com/vllm-project/vllm) 和 [SGLang](https://github.com/sgl-project/sglang)，多卡训练用 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 和 [DeepSpeed](https://github.com/microsoft/DeepSpeed)，强化学习用 [veRL](https://github.com/volcengine/verl) 和 [Slime](https://github.com/THUDM/slime)。每一个分布式框架都维护着自己的一套模型实现——同一个模型在生态里同时存在多份互相独立、需要各自维护的代码；各框架的优化同样与各自的模型实现绑定，无法跨框架复用。

本文提出的命题是：**基于 DTorch 的易用性，将 PyTorch 分布式生态的技术底座由 Multi-Controller + SPMD 替换为 Single-Controller + DTensor，使整个分布式生态回归 single-device 时代的形态**——模型实现只有一份，分布式仅是 DeviceMesh 与 Placements 的配置。

文章结构如下：第 2 节分析生态碎片化的现状、成因、代价与本质；第 3 节给出重构方案（Single-Controller + DTensor 替换 Multi-Controller + SPMD）并展开论证；第 4 节分析重构后各框架的机遇；第 5 节给出实现路径与开发成本分析；第 6 节讨论挑战；第 7 节总结。

## 2 PyTorch 分布式生态现状

### 2.1 各框架分工

按任务类型划分，当前 PyTorch 生态的格局如表 1 所示：

<figure markdown>
  |场景|框架|做法|
  |-|-|-|
  |single-device 训练与推理|[transformers](https://github.com/huggingface/transformers)、[diffusers](https://github.com/huggingface/diffusers)、[trl](https://github.com/huggingface/trl)|提供一份易用的模型与算法实现，是生态中事实上的“模型标准库”|
  |多卡推理|[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)|各自重写一套主流模型的并行实现，并叠加推理服务优化（PagedAttention、continuous batching 等）|
  |多卡训练|[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[DeepSpeed](https://github.com/microsoft/DeepSpeed)|Megatron-LM 提供专用并行层并要求模型改写；DeepSpeed 侧重 ZeRO 等优化，TP / PP 仍需 Megatron 风格的模型适配|
  |强化学习|[veRL](https://github.com/volcengine/verl)、[Slime](https://github.com/THUDM/slime)|训练引擎与推理引擎协同工作（引入 Ray 构成混合范式），并处理两者间的权重同步|

  <figcaption>表 1：PyTorch 生态按任务类型的碎片化格局</figcaption>
</figure>

它们各自解决不同的问题，但除 single-device 的模型实现之外，**每一个分布式框架都各自维护着一套模型实现**。这提高了适配的门槛与时间成本：分布式代码的实现难度高，需要框架开发人员完成适配，算法工程师很难自己动手；跨框架协作还需要写大量的胶水代码，弥合分布式与单机、各框架之间的差异。算法工程师最终无法像在单卡上那样快速实现并迭代算法。

### 2.2 碎片化的成因：按需生长的演进历史

碎片化不是某个组织设计出来的，而是按需生长出来的。回顾 PyTorch 分布式接口的演进历程（详见 [《DTorch 介绍》3.2.2 节「演进历史」](https://tingkuanpei.github.io/dtorch/cn/blog/introduction/#322-演进历史)），每一个阶段的新需求都催生了一个新框架，如表 2 所示：

<figure markdown>
  |阶段|需求|生态的回应|
  |-|-|-|
  |单卡时代（2016~）|单卡训练与推理|PyTorch eager + HuggingFace transformers / diffusers / trl，模型实现集中在一处|
  |DDP（2018）|Data Parallel|PyTorch DDP（SPMD 范式），无需修改模型代码|
  |TP / PP / EP（2019~2022）|大模型训练|Megatron-LM 登场，模型需要改写、权重需要预先切分|
  |LLM 推理（2023~）|高吞吐推理服务|vLLM、SGLang 各自重写主流模型实现并叠加推理优化|
  |强化学习（2024~）|RLHF / GRPO 训练|veRL、Slime 登场，训练与推理引擎协同（Ray 混合范式）|

  <figcaption>表 2：PyTorch 分布式生态的演进历史——每个阶段的新需求催生一个新框架</figcaption>
</figure>

除 DDP 阶段外，每一个新阶段都意味着“再维护一套模型实现”。原因在于：PyTorch 的分布式接口从 DDP 开始沿用至今，从未有过统一的设计（[《DTorch 介绍》3.2.2 节「演进历史」](https://tingkuanpei.github.io/dtorch/cn/blog/introduction/#322-演进历史)）——不需要修改模型代码的只有 Data Parallel 和 ZeRO，其余并行方式都需要定制化修改。于是每个新需求出现时，社区没有现成的“分布式 API”可用，只能新建一个框架、把模型再实现一遍。

### 2.3 碎片化的代价

碎片化的代价直接落在每一个模型、每一个用户身上，如表 3 所示：

<figure markdown>
  |代价|说明|
  |-|-|
  |同一模型多份实现|HuggingFace 单卡版 + Megatron 训练版 + vLLM 推理版，功能与修复需要在多套实现之间同步|
  |新模型落地慢|每个框架各适配一遍，社区往往需要数周才能完成一个模型的分布式适配|
  |权重格式转换|训练与推理引擎的切分方式、参数命名不同，需要 checkpoint conversion 工具链|
  |训推数值不一致|两套 kernel 实现，同一份权重的输出存在数值差异，在强化学习中代价最大|
  |算法与工程割裂|分布式代码只能由经验丰富的工程人员魔改，算法人员使用魔改后的结果（[链接](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/#3-接入新的算法更方便)）|
  |调试成本高|多进程日志混排、断点导致集合通信 hang、死锁难以定位（[链接](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/#4-调试与可观测性)）|

  <figcaption>表 3：分布式生态碎片化的代价</figcaption>
</figure>

### 2.4 本质：分布式 API 层缺位

透过碎片化的表象，其本质是一个层次结构的问题。一个深度学习生态可以分成三层：

- **算子库层**：提供 Tensor 与算子（PyTorch 算子、FlashInfer、FlashAttention 等）；
- **分布式 API 层**：回答“一个 Tensor 如何被切分、计算如何在多设备上编排”；
- **解决方案层**：vLLM、Megatron-LM、veRL 等，解决具体领域的问题。

**分布式 API 层是缺位的。**PyTorch 目前在分布式上提供的是 SPMD 方案——ProcessGroup、集合通信等底层原语：只有进程与通信的机制，切分、通信、调度全部由用户承担，本质上等于没有方案——除 Data Parallel 和 ZeRO 外，其余并行方式均需定制化修改。

于是，每一个解决方案都必须自己重造这一层：vLLM 自研了 `QKVParallelLinear` 等并行层，Megatron-LM 维护着 `ColumnParallelLinear` / `RowParallelLinear` 等并行层——**同样的工作被每个框架重复做了一遍，而这套重复的并行层又绑定了各自的模型实现**。碎片化由此而来。

反过来说，如果分布式 API 层被补上，解决方案层就无需再各自重造——这正是 DTorch 要做的事。

## 3 重构方案

当前 PyTorch 生态在分布式层的技术底座是 **Multi-Controller + SPMD**：每块 GPU 一个进程，每个进程既是控制者又是执行者，代码以 rank 为中心、切分与集合通信全部手动完成。[第 2.4 节](#24-本质分布式-api-层缺位) 指出的 API 层缺位，其根源正在这套底座。

重构方案：**用 Single-Controller + DTensor 替换 Multi-Controller + SPMD 这套技术底座，模型实现层不动**——在 HuggingFace 的 [transformers](https://github.com/huggingface/transformers)、[diffusers](https://github.com/huggingface/diffusers)、[trl](https://github.com/huggingface/trl) 的基础上增加 DTensor 支持，开发体验回归 single-device 时代。

### 3.1 支柱一：Single-Controller，程序不用多进程

Multi-Controller 下，程序由 torchrun 启动 N 个进程，代码处处以 rank 为中心，多进程的开发与调试成本——日志混排、断点导致集合通信 hang、死锁难以定位——全部由用户承担（[《DTorch 的优势与机遇》第 4 节「调试与可观测性」](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/#4-调试与可观测性)）。

Single-Controller 下，全集群只有一个控制者、统一管理全部 GPU：用户程序是单线程的，以全局视角描述计算——没有进程、没有 rank、没有 torchrun，切分、通信与调度全部由框架隐式完成，代码接近单卡程序。

### 3.2 支柱二：DTensor，模型代码不用改

在 single-device 程序上配置 DTensor 的 DeviceMesh 和 Placements 即可完成对分布式的支持。各并行方式在 Module 层的用法见 [Module 并行](https://tingkuanpei.github.io/dtorch/cn/user_guide/module_parallel/)，DP + TP + PP + CP 任意组合与切换的示例见 [Llama 并行示例](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/)。

### 3.3 极致的易用性：回归 single-device 时代

Single-Controller + DTensor 两个支柱叠加的效果，是**极致的易用性**：

- **算法工程师**可以按照自己的想法自由实现模型结构，并扩展到多机多卡，完成分布式训练和推理——不再需要阅读、定制并调试晦涩难懂的 SPMD 代码；
- **框架开发者**摆脱 SPMD 造成的繁琐开发工作，把时间投入到算子优化、系统优化等真正有价值的工作上。

整体的开发体验，回归 single-device 时代。回顾深度学习框架史，PyTorch 的性能往往并不占优，却凭借易用性成为事实标准。而在分布式时代，易用性这块高地尚无人占据——同样的故事，正等待再次上演。

## 4 重构后各框架的机遇：解放而非取代

替换为 Single-Controller + DTensor 方案后，模型实现回归 HuggingFace 提供的标准实现，现有分布式框架不必再各自维护一套并行模型实现。但重构对它们而言不是取代，而是赋能——DTorch 补上的分布式 API 层替它们接过了分布式执行的重担，各框架的出路随之清晰：

**借助 DTorch 易用的分布式 API，框架从繁琐的分布式多进程调度与管理中解放出来，把时间投入到算子级/系统级优化这类更有价值的工作上**。vLLM 和 SGLang 中的 PagedAttention、调度与融合策略、Megatron 的通信优化，都是各框架多年打磨的核心成果；过去它们必须依附于一整套模型重写才能交付，如今只需专注优化本身，直接应用到 HuggingFace 的标准模型实现上。**所有的优化都变成可自由组合的算子，不再和任何具体的框架绑定，任何模型都能使用**。

框架之间的竞争焦点随之转移：从“谁的模型实现更全”转移到“谁的算子与系统优化更深”。对用户而言，模型实现不再与任何框架绑定，**用户从生态重构中获得的正是这份选择自由**——训练和推理时可以选择最好的算子组合，而不必把整个模型迁到某个框架里。

## 5 实现路径和开发成本分析

### 5.1 实现路径

生态重构按照先易后难、逐步加大投入的方式推进，每一步都有明确的落点：

1. **Diffusion 推理**：diffusers 库增加 DTensor 支持，扩散模型即可在多机多卡上完成分布式推理。DTorch 已基于 `diffusers==0.34.0` 迁移了 SD3 / FLUX 两款扩散模型作为示范；
2. **LLM 推理**：transformers 增加 DTensor 支持——改动与 diffusers 的示范一致——并补齐 PagedAttention、continuous batching 等推理服务优化，性能达到 vLLM 的水平；
3. **完善对训练组件的支持**：除 autograd 与优化器之外，DP / PP / EP / ZeRO 等并行方式都需要专门支持（如训练中的计算与通信重叠），逐一完成后，transformers / trl 的分布式训练随之可用；
4. **完成对 RL 训练的支持**：训练与推理本就是同一执行引擎的两种使用方式，多角色编排由 Single-Controller 原生完成，veRL / Slime 覆盖的场景随之统一编排。

### 5.2 开发成本分析

**分项人力成本**：预计 10 人团队即可完成推理相关功能，15 人团队即可完成训练相关功能。

**总成本对比**：重构是一次性成本——框架能力与 HuggingFace 库接入各自只需投入一次；而现状下的模型适配成本由每个框架重复支付（N 个框架 × M 个模型，且随新模型发布持续产生）。一边是一次性投入，一边是随模型数量持续膨胀的重复投入，生态总开发成本的差距一目了然。

## 6 挑战

重构方案要成立，以下挑战必须正面回应：

- **DTorch 的训练和推理能力尚不成熟**（[《DTorch 的优势与机遇》第 10 节「劣势」](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/#10-劣势)）：目前已落地的只有扩散模型推理示范，[第 5.1 节](#51-实现路径)的其余步骤——LLM 推理、训练组件、RL 训练——都还是路线图；
- **DTorch 解决的只是分布式的问题**：Single-Controller + DTensor 解决“并行”与“易用性”，但算子优化、请求调度、计算与通信重叠等优化是推理与训练框架多年积累的成果，并不自动随之而来，重构后仍需逐一接入；
- **生态惯性**：现有框架的用户与维护者需要付出迁移成本，vLLM / Megatron 的成熟度与社区积累不会一夜蒸发；

DTorch 的定位应当是解决方案，而非止步于框架：主动落地 Diffusion 推理、LLM 推理、RL 训练等解决方案来推广 Single-Controller + DTensor 范式，而不是被动等待其他框架接入。一旦对应的功能完成，DTorch 就能凭借易用性吸引算法工程师使用，并在新模型的适配上展现优势。

## 7 总结

PyTorch 分布式生态碎片化的本质，是分布式 API 层缺位。DTorch 基于 Single-Controller + DTensor 的方案实现了易用的分布式 API。基于 DTorch，分布式训练和推理框架可以从繁琐的多进程的调度和管理中释放出来，开发体验和易用性回到 single-device 时代。

回顾深度学习框架史，PyTorch 的性能往往并不占优，却凭借易用性成为事实标准。而在分布式时代，易用性这块高地尚无人占据——同样的故事，正等待再次上演。

## 延伸阅读

- [《DTorch 介绍》](https://tingkuanpei.github.io/dtorch/cn/blog/introduction/) — DTorch 与 PyTorch 的全面对比、API 简介、精度与性能数据
- [《DTorch 架构设计：简洁与高效何以兼得》](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/) — 三大核心设计详解
- [《DTorch 的优势与机遇》](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/) — 优势、行业机会、劣势与路线图
