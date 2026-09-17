# DTorch：重构 PyTorch 生态，迈向 GPU 集群操作系统

> **阅读提示：文中会用到 Distributed Tensor、Single-Controller 和 Multi-Controller 等概念，不熟悉的读者可先参考：[Distributed Tensor Overview](https://tingkuanpei.github.io/dtorch/cn/user_guide/distributed_tensor_overview/)、[Single-Controller 与 Multi-Controller](https://tingkuanpei.github.io/dtorch/cn/developer_guide/single_and_multi_controller/)。DTorch 和 PyTorch 分布式的代码对比请参考：[Github 页面](https://github.com/tingkuanpei/dtorch#example)**

## 1 概述

当前 PyTorch 为支持分布式计算提供了多种范式，不同的训练和推理框架选用不同的范式组合，如[表 2](#table-2)所示。**其中，Native Tensor（单卡上的 Tensor） + Multi-Controller(SPMD)的方案应用最为广泛，但其使用繁琐，门槛高，易用性极差。**Multi-Controller(SPMD)范式要求 Python 代码直接驱动本机的 GPU，调度过程对外是一个黑盒。集群管理系统无法感知容器内执行的计算，也无法高效迁移 GPU 容器([§6.6](#66-现有的容器迁移方案))，这导致了集群的 GPU 平均利用率普遍不高<sup><a href="#note-1">注释1</a></sup>——GPU 一经分配，利用率再低也无法收回再调度；任一机器故障，整个任务随之重启 GPU。

为了解决上述问题，DTorch 完全抛弃 SPMD，**基于 Single-Controller + DTensor 构建了简洁、易用且完备的分布式 API**：用户不再需要编排进程与通信，分布式编程回到 single-device 时代，示例代码如[表 1](#table-1)所示。**这套 API 同时是 GPU 集群操作系统的标准接口，如[图 1](#image-1)所示——集群由此感知计算内容，统一调度存储、计算与网络资源，构成易用、低成本、高性能和高可靠性的 GPU 集群**。全球每年 AI/GPU 服务器硬件的市场规模已达数千亿美元，行业急需能感知计算内容、统一调度集群资源的 GPU 集群操作系统，谁先建成，谁就掌握下一代生态的底座。虽然重构 PyTorch 的生态需要投入大量的成本，但是集群利用率只要提高 1%，节省的成本就远超重构生态所需的全部投入。

本文结构如下：第 2 章分析 PyTorch 分布式生态碎片化的现状、成因、本质与代价；第 3 章介绍 DTorch 的分布式 API；第 4、5 章论证 DTorch API 的两块基石——Why DTensor 与 Why Single-Controller；第 6 章分析 GPU 集群的困境，并展开 GPU 集群操作系统的设想；第 7 章描绘重构后的生态图景，并给出实现路径、劣势与挑战；第 8 章总结。

<figure markdown id="image-1">
  ![DTorch API：GPU 集群操作系统的标准接口](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@latest/blog/gpu_cluster_os_cn.png)
  <figcaption>图 1：GPU 集群操作系统示意图<br> 用户在 Python Client 上描述计算逻辑，计算逻辑会被序列化为 Operator 消息队列，发送到集群上异步执行。集群统一管理所有的 GPU Worker，可以高效调度</figcaption>
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
mesh = dtorch.DeviceMesh("cuda", [0, 1])
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
<figcaption>表 1：DTorch 和 PyTorch 的写法对比 <br> 创建并打印在第 0 维被切分的分布式 Tensor，DTorch 程序是单线程的，不需要考虑每个 rank 的行为</figcaption>
</figure>

## 2 PyTorch 分布式生态

本章先呈现 PyTorch 分布式生态碎片化的现状，再回顾其按需生长的演进历史，继而透过表象指出本质——分布式 API 层缺位。

### 2.1 现状：碎片化的生态格局

当前 PyTorch 为支持分布式计算提供了多种范式：张量表示有 Native Tensor（单卡上的 Tensor）与 DTensor(Distributed Tensor)；控制范式有 Multi-Controller(SPMD)与 Single-Controller + Multi-Controller(SPMD)。相关概念请参考：[Distributed Tensor](https://tingkuanpei.github.io/dtorch/cn/user_guide/distributed_tensor_overview/)、[Single-Controller 与 Multi-Controller](https://tingkuanpei.github.io/dtorch/cn/developer_guide/single_and_multi_controller/)。不同的训练和推理框架选用不同的范式组合，如[表 2](#table-2) 所示。这些组合在成熟、完备与易用上各有短板，没有一种能三者兼得：

  - Native Tensor 需要手动推导编排 Tensor 在所有 rank 上的行为，在合适的地方插入通讯算子，打印 Tensor 时需要手动聚合各个 rank 的值。
  - SPMD 范式要求所有 rank 执行同一份代码，rank 间行为不一致时只能以 if-else 区分执行路径；而 TP、PP、EP 等并行方式以及 RL 训练流程中，不同 rank 的行为差异极大，PyTorch 分布式代码的易用性因此急剧恶化。
  - Native Tensor + Multi-Controller(SPMD)的方案应用最为广泛，但其使用繁琐，门槛高，易用性极差。
  - DTensor 的方案开发多年，至今仍处于 [alpha state and under development](https://docs.pytorch.org/docs/stable/distributed.tensor.html)。
  - 在分布式系统中，vllm 为解决推理请求的调度问题，verl 为解决 RL 流程的多角色异构编排问题, 它们还引入 ray 管理跨机器的进程，组成 Single-Controller + Multi-Controller(SPMD) 的混合范式，进一步提高了使用门槛。

按照任务类型划分，则如[表 3](#table-3)所示：同一个模型在不同框架中存在多份互相独立的实现，优化又与框架的实现绑定，无法跨框架复用。再往下细分到 DP、TP、PP、CP、EP 和 ZeRO 等具体并行方案，各框架和范式支持的范围与深度更是参差不齐。

在这样的生态中，分布式代码的实现难度高，模型适配依赖经验丰富的框架开发人员；跨框架协作还需要大量胶水代码，弥合分布式与单机、各框架之间的差异。算法工程师被挡在分布式之外，无法像在单卡上那样快速实现并迭代算法。

<figure markdown id="table-2">
  |张量表示＼控制范式|Multi-Controller(SPMD)|Single-Controller + Multi-Controller(SPMD)|
  |-|-|-|
  |Native Tensor|Megatron-LM、DeepSpeed、SGLang|vLLM、veRL、Slime、Monarch|
  |Distributed Tensor|TorchTitan、PyTorch FSDP2|  |

  <figcaption markdown="span">表 2：生态中各框架按“张量表示 × 控制范式”的落位，相关概念请参考：[Distributed Tensor](https://tingkuanpei.github.io/dtorch/cn/user_guide/distributed_tensor_overview/)、[Single-Controller 与 Multi-Controller](https://tingkuanpei.github.io/dtorch/cn/developer_guide/single_and_multi_controller/)</figcaption>
</figure>

<figure markdown id="table-3">
  |场景|框架|做法|
  |-|-|-|
  |single-device 训练与推理|transformers、diffusers、trl|提供一份易用的模型与算法实现，是生态中事实上的“模型标准库”|
  |多卡推理|vLLM、SGLang|各自重写一套主流模型的并行实现，并叠加推理服务优化（PagedAttention、continuous batching 等）|
  |多卡训练|Megatron-LM、DeepSpeed|Megatron-LM 提供专用并行层并要求模型改写，TP / PP 仍需 Megatron 风格的模型适配；DeepSpeed 侧重 ZeRO 等优化，|
  |强化学习|veRL、Slime|训练引擎与推理引擎协同工作（引入 Ray 构成混合范式），由于训练和推理框架模型实现不同，因此需要处理两者间的权重转换和同步|

  <figcaption>表 3：PyTorch 生态按任务类型划分</figcaption>
</figure>

### 2.2 碎片化的成因：按需生长的演进历史

碎片化不是某个组织设计出来的，而是按需生长出来的。回顾 PyTorch 分布式接口的演进历程（详见 [《DTorch 介绍》3.2.2 节「演进历史」](https://tingkuanpei.github.io/dtorch/cn/blog/introduction/#322-演进历史)），每一个阶段的新需求都催生了一个新框架，如[表 4](#table-4)所示：

<figure markdown id="table-4">
  |阶段|需求|生态的回应|
  |-|-|-|
  |单卡时代（2016~）|单卡训练与推理|PyTorch eager + HuggingFace transformers / diffusers / trl，模型实现集中在一处|
  |DDP（2018~）|Data Parallel|PyTorch DDP（SPMD 范式），无需修改模型代码|
  |TP / PP / EP（2019~）|大模型训练|Megatron-LM 基于 SPMD 范式实现 TP、PP、EP 等并行方案。模型需要改写、权重需要预先切分|
  |LLM 推理（2023~）|高吞吐推理服务|vLLM、SGLang 基于 SPMD 范式实现 TP、PP、EP 等并行方案，各自重写主流模型实现并叠加推理优化|
  |强化学习（2024~）|RLHF / GRPO 训练|veRL、Slime 引入 Ray 做中心调度，协同训练与推理引擎（混合范式），并处理两者间的权重同步|

<figcaption>表 4：PyTorch 分布式生态的演进历史</figcaption>
</figure>

PyTorch 的分布式接口从 DDP 开始沿用至今，从未有过统一的设计。每当新需求出现，社区没有现成的分布式 API 可以复用，只能新建框架、把模型重新实现一遍——这是生态碎片化的直接成因。

### 2.3 本质：分布式 API 层缺位

PyTorch 分布式生态碎片化，是因为其缺乏完备、易用且统一的分布式 API。把一个分布式深度学习框架的生态按层次拆开，可以分为三层：

- **算子库层**：提供高性能的算子实现（PyTorch 算子、FlashInfer、FlashAttention 等）；
- **分布式 API 层**：解决 Tensor 如何被切分、计算如何在多设备上编排的问题；
- **解决方案层**：vLLM、Megatron-LM、veRL 等，解决具体领域的训练和推理问题。

**PyTorch 的分布式 API 层长期缺位**，其只提供 SPMD 范式下的进程创建、ProcessGroup 集合通信等底层功能，Tensor 的切分、通信与调度全部交给用户。而用户迫切需要的 DTensor 与 Single-Controller 方案，则因与既有 SPMD 范式存在本质冲突而进展缓慢：DTensor 开发多年，至今仍处于 [alpha state and under development](https://docs.pytorch.org/docs/stable/distributed.tensor.html)；Single-Controller 只能借助 Ray 或 Monarch 创建并管理远端进程，以 Single-Controller + Multi-Controller(SPMD)的混合形态存在。

每一个训练和推理框架都只能自己定义并实现自己的分布式 API 层，再基于它去支持并行计算，做算子优化和 pipeline 优化。各框架的实现互不兼容、无法组合。碎片化的代价由每一个模型、每一个用户反复支付——每一次版本发布、每一个新模型、每一轮实验，都要重新支付一次，如[表 5](#table-5)所示：

<figure markdown id="table-5">
  |代价|说明|
  |-|-|
  |同一模型多份实现|HuggingFace 单卡版 + Megatron 训练版 + vLLM 推理版，功能与修复需要在多套实现之间同步|
  |新模型落地慢|每个框架各适配一遍，社区往往需要数周才能完成一个模型的分布式适配|
  |权重格式转换|训练与推理引擎的切分方式、参数命名不同，需要 checkpoint conversion 工具链|
  |训推数值不一致|两套 kernel 实现，同一份权重的输出存在数值差异，在强化学习中代价最大|
  |算法与工程割裂|分布式适配依赖资深的框架开发人员，算法人员无法自己动手，新想法要等适配完成才能在集群上验证|
  |调试成本高|Tensor在多卡上的分布方式需要自行推导；多进程日志混排；无法断点调试(断点导致集合通信 hang、死锁难以定位)|

  <figcaption>表 5：分布式生态碎片化的代价</figcaption>
</figure>

## 3 DTorch：易用的分布式 API

为解决 PyTorch 分布式 API 易用性差、GPU 集群利用率低、可靠性差的问题，DTorch 基于 Single-Controller 和 DTensor（Distributed Tensor）设计了一套简洁、易用且统一的分布式深度学习 API。DTorch 和 PyTorch 示例代码的对比如[表 1](#table-1)所示。这套 API 的核心能力包括：

- **单线程编程**：**用户在单线程中以全局视角描述计算，进程编排、通信与同步全部由框架完成，分布式编程回到 single-device 时代，开发与调试体验同单卡程序一致**；
- **并行只是配置**：同一份模型代码无需修改，即可运行在单机或 DP、TP、PP、CP 等并行方案及其组合下，并行度与切分方式由 DeviceMesh 与 Placements 描述，切换并行方案只需调整配置；
- **优化自由组合**：量化、Cache、Kernel 融合等优化与并行方案正交，可在同一份模型实现上按需叠加；强化学习中各角色的编排只是 DeviceMesh 配置的差异，角色间的权重传输只是普通的 redistribute 操作，训练与推理又共用同一份模型实现——[表 5](#table-5) 所列的权重转换、权重同步与训推数值不一致等代价随之消失；
- **更低的 CPU 开销**：Client、Controller 与 Worker 异步执行，Python Client 的执行时间比 PyTorch 低约 64%，CUDA Kernel launch 更及时，模型端到端时延反而略降，SD3 实测数据见[表 6](#table-6)。

DTorch 的文档与此前的博客已对此做过系统介绍：

- **DTorch 的整体情况与 API 介绍，以及精度和性能测试数据，可参考：[DTorch 介绍](https://tingkuanpei.github.io/dtorch/cn/blog/introduction/)**；
- DTorch 的架构设计方案，包括如何以重叠的方式解决 Single-Controller 的调度开销、如何支持 DTensor、如何以 Eager Graph 架构统一动态图与静态图，可参考：[架构设计](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/)；
- 如何在 Module 层实现各类并行方案，让同一份代码同时支持单机和分布式模式，可参考：[Module 并行](https://tingkuanpei.github.io/dtorch/cn/user_guide/module_parallel/)；
- 在 Llama 模型上开启各类并行方案的完整示例，可参考：[Llama 并行示例](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/)。

第 4 章和第 5 章将分别介绍为什么选择 DTensor 和 Single-Controller 方案构建 DTorch API。第 6 章将介绍如何基于 DTorch API 构建 GPU 集群操作系统，统一管理和调度集群的存储、计算与网络资源。

## 4 Why DTensor

DTorch 基于 Single-Controller 和 DTensor 设计了一套易的分布式 API。本章将阐述为什么选择 DTensor 方案。

DTensor 在 Native Tensor 的基础上增加了 DeviceMesh 和 Placements 属性，用来描述 Tensor 在多卡上如何切分与存储。Native Tensor 不携带分布信息，切分、通信、聚合只能由用户手动编排，由此产生大量冗余代码；DTensor 把分布信息放进 Tensor 之后，这些工作全部由框架自动完成：

- **Tensor 自带全局描述**：Native Tensor 如何在多卡上分布、Global Shape 是什么，只能由用户根据上下文手动推导；DTensor 的 Shape 即 Global Shape，分布由 DeviceMesh 与 Placements 显式描述，打印即可获知；
- **加载权重自动切分**：加载一份完整的 state_dict，框架按 Placements 自动切分到各 rank，无需像 Megatron-LM 那样按 TP / PP rank 预先切分和转换权重；
- **取值自动聚合**：`to_torch()` 自动把各 rank 的分片聚合回完整 Tensor，无需手动 all-gather；
- **通信即算子**：`redistribute()` 是一个普通算子，用户无需创建和管理 ProcessGroup，也无需手动调用 all-reduce 等集合通信；框架在底层根据计算图与网络拓扑选择最高效的实现，并让计算与通信自动重叠；
- **不均匀切分自动处理**：维度长度不能被设备数整除时自动计算本地 shape，通信时自动 padding / unpadding，全程对用户透明；

更重要的是，DTensor 将 DP、TP、PP、CP、EP、ZeRO 等并行方式统一表达为 DeviceMesh 与 Placements 的组合——**同一份模型代码，配置不同的 DeviceMesh 与 Placements，即可组合开启任意并行方式，并行策略从模型实现退化为一份配置**（各并行方式在 Module 层的用法见 [Module 并行](https://tingkuanpei.github.io/dtorch/cn/user_guide/module_parallel/)，DP + TP + PP + CP 任意组合与切换的示例见 [Llama 并行示例](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/)）。

## 5 Why Single-Controller

DTorch 基于 Single-Controller 和 DTensor 设计了一套易的分布式 API。本章将阐述为什么选择 Single-Controller 方案。

Single-Controller 与 Multi-Controller + SPMD 的取舍并非新问题：2015 年的 TensorFlow v1 正是 Single-Controller——由单一 client 构建计算图、驱动整个集群执行，但受限于当年的实现，中心化调度给行业留下了“中心化调度 = 慢”的印象。2018 年前后，all-reduce 与 DDP 走向成熟，PyTorch 彻底倒向 Multi-Controller + SPMD。这一选择在当时是成立的：

- Native Tensor 只携带局部数据，每个进程都看不见全局，最自然的做法是让每个进程各自管理自己、用 rank 和集合通信相互协调，SPMD 与 Data Parallel 也天然契合；
- Controller 与 GPU 同机，调度只经过 PCIe 总线，没有中心化调度的开销。

但大模型时代打破了这两个前提。其一，SPMD 要求所有 rank 执行同一份代码，rank 间行为不一致时只能以 if-else 区分执行路径；而 TP、PP、EP 等并行方式以及 RL 训练流程中，不同 rank 的行为差异极大，PyTorch 分布式代码的易用性因此急剧恶化。DTensor 携带全局数据描述，能简洁地表达各种并行方式，与 Single-Controller 的全局视野天然契合——DTensor + Single-Controller 才是大模型时代更自然的组合。其二，Single-Controller 的调度开销已被 DTorch 的异步流水线摊销，实测与 Multi-Controller 基本相当（见 5.1 节）。

两个前提至此全部失效，Single-Controller 成为更自然的选择。本章接下来逐一展开：5.1 节论证 Single-Controller 的调度开销已被摊销，5.2 节展示 Single-Controller 在易用性上的优势，5.3 节介绍全局视野带来的系统能力，5.4 节以 Monarch 为例介绍 PyTorch 的 Single-Controller 方案，5.5 节小结。

### 5.1 调度开销已经不是问题

历史上，行业拒绝 Single-Controller 的主要理由是调度开销：Multi-Controller 的 Controller 与 GPU 同机，调度仅需经过 PCIe 总线；Single-Controller 调度远端 GPU 需经过跨机网络通信。

Single-Controller 的代价被过度放大了——TensorFlow v1 慢的根源不在“中心化”本身，而在于调度无法与计算并行：调度耗时直接加到每一步的时延上。TensorFlow v1 的执行以 `session.run` 为单位：Client 每次调用都同步阻塞，等待本轮 step 的全部计算完成；Master 把切分好的子图整体下发给各 Worker 执行，并非逐算子派发。即便如此，step 之间仍是串行的——下一步的调度无法与当前 step 的计算重叠（输入预取等 IO 环节除外）；而若将中心化调度细化到逐算子派发，每个算子的调度决策都需要一次跨机往返，更加不可行。无论哪种形态，调度与计算都只能串行进行。

DTorch 通过重叠调度和计算，解决了 Single-Controller 的调度开销。深度学习程序由一系列 Tensor 和 Operator 组成，而 Operator 输出 Tensor 的数据类型、Shape 等元信息，在计算完成之前可以确定。框架因此可以在前序 Kernel 计算的同时，完成后续算子的构图与调度，使构图、调度与 Kernel 计算相互重叠（overlap）。DTorch 在实现上采用 Single-Client Single-Controller Multi-Worker 三级异步流水线，详见[《DTorch 架构设计：简洁与高效何以兼得》](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/)。

DTorch 的实测数据也支持这一结论，如[表 6](#table-6)所示：

<figure markdown id="table-6">
  |层面|PyTorch|DTorch|说明|
  |-|-|-|-|
  |小算子端到端（`Shape=(1,)` 的 add 算子）|5.93us|8.14us<span style="color: #ff9800; font-weight: bold;">（+37%）</span>|随 Shape 变大，差距趋近于零；小算子时延仍有优化空间|
  |大算子端到端（SDPA 算子）|基准|基本一致|算子计算占主导|
  |SD3 Python Client 执行时间|0.575s|0.206s<span style="color: #4caf50; font-weight: bold;">（-64.17%）</span>|异步执行省下的 CPU 时间，用于 overlap 系统调度开销|
  |SD3 单卡推理端到端耗时|1.683s|1.648s<span style="color: #4caf50; font-weight: bold;">（-2.08%）</span>|CUDA Kernel launch 更及时且密集，时延反而略降|
  |SD3 峰值显存|18.131GB|17.663GB<span style="color: #4caf50; font-weight: bold;">（-2.58%）</span>|异步执行下中间 Tensor 及时释放|

  <figcaption>表 6：DTorch 与 PyTorch 的调度开销与性能对比</figcaption>
</figure>

### 5.2 最易用的分布式 API

**DTorch 的分布式 API 提供与 single-device 时代一致的开发与调试体验。** Single-Controller 把整个分布式集群抽象成一台大型计算机：用户在单线程中编程，即可调用集群的全部计算资源——没有 torchrun、没有多进程、没有 ProcessGroup，无需区分每个 rank 的执行路径，以全局视角描述计算。调试同样回到单卡时代：直接打断点、打印 Tensor 的值；不再有日志混排、多进程 race condition、集合通信导致的死锁；还可以在单卡上模拟多卡环境，完成分布式代码的开发与调试。

Single-Controller 还解决了一类 SPMD 难以处理的问题——异构角色的编排。强化学习训练（PPO / GRPO）是最典型的场景：actor 生成、reference model 与 reward model 推理、参数更新各自占用一组 GPU，交替执行且依赖关系复杂。Multi-Controller 生态中，veRL 为此引入 Ray 做中心调度，构成 Single-Controller + Multi-Controller 的混合范式，并专门解决训练引擎与推理引擎之间的权重同步。DTorch 原生即为 Single-Controller：各角色使用不同的 DeviceMesh 即可——这只是配置层面的差异，不涉及代码逻辑的修改；所有角色由同一个 Controller 编排，角色之间的数据传输（激活、权重）是普通的 DTensor 操作（`redistribute`），无需跨引擎的格式转换与同步协议。

### 5.3 全局视野带来的系统能力

Single-Controller 持有全局计算图与全局设备状态，一系列在 Multi-Controller 的 SPMD 范式下需要大量手动协调、甚至无法实现的能力成为可能：

- **自动通信与调度**：`redistribute()` 是普通算子，框架自动选择最高效的通信实现、插入必要的同步节点（CUDA Event），并让计算与通信重叠；通信死锁在构图时即可被检测和规避。用户无需创建和管理 ProcessGroup，也无需关心 all-reduce 应该插在哪里。
- **全局设备管理与设备虚拟化**：DeviceMesh 只是一组 Device ID 的描述，物理设备由框架统一管理，因此可以把多个虚拟设备映射到同一物理设备上。
- **故障设备驱逐与自动恢复**：Controller 捕获设备故障抛出的异常，将任务迁移到健康的设备上继续执行，避免整个作业失败重启。
- **运行时动态增删计算节点**：Controller 统一管理所有 Worker，可以在程序运行过程中向集群加入新的计算节点或将其移除，实现集群的弹性伸缩——SPMD 的进程组在 torchrun 启动时便已固定，运行期扩缩容需要重组进程组并重启。
- **Auto Parallel 与 JIT 编译**：依靠全局信息进行自动并行策略搜索与编译优化。

### 5.4 Monarch：PyTorch 的 Single-Controller 方案

PyTorch 团队同样意识到了 Single-Controller 方案的价值，并开发了原生支持 Single-Controller 的 [Monarch](https://github.com/meta-pytorch/monarch) 项目。Monarch [博客](https://pytorch.org/blog/introducing-pytorch-monarch/)介绍了它的两个核心功能：

> - 可扩展消息的远端 Actor：Actor 被组织为称为 mesh 的集合，消息可以广播给所有成员。
> - 基于 supervision tree 的容错：Actor 与进程构成一棵树，故障沿树向上传播，提供良好的默认错误处理行为，并支持细粒度的故障恢复。

Monarch 的核心功能与 Ray 基本一致：以更便捷的方式（Actor）创建和管理远端进程，并支持错误处理机制。本质上，Monarch 仍是 Single-Controller + Multi-Controller(SPMD)的混合范式——其优势是兼容性好，可以复用 Megatron-LM、SGLang 中已有的代码；劣势是把多进程直接暴露给用户，在 PP、EP 等不同 rank 行为差异大的并行方式下，易用性依旧很差。

Monarch 也意识到了上述问题，提出了 [Distributed Tensors in Monarch](https://meta-pytorch.org/monarch/stable/generated/examples/distributed_tensors.html) 方案。但这里的 Distributed Tensors 与[第 4 章](#4-why-dtensor)的 Distributed Tensors 不同：它没有 DeviceMesh 和 Placements 属性，可以理解为同时运行在多个 device 上的 Native Tensor。这个方案不兼容 PyTorch SPMD 生态现有的实现，调度开销预计也很大，但仍是一种积极的探索。

### 5.5 小结

Single-Controller 是分布式系统下一步的发展方向：vLLM 与 veRL 引入 Ray 做中心调度，PyTorch 提出了 Monarch——行业已经反复用行动证明了这一点。PyTorch 生态为兼容已有的分布式代码，保留 SPMD，构成 Single-Controller + Multi-Controller(SPMD)的混合范式；DTorch 则从易用性出发，彻底抛弃 SPMD，只兼容 single-device 的 PyTorch 代码，构建最易用的分布式 API。

极致的易用性让 DTorch 获得了生存空间，但还不足以构成对 PyTorch 生态的绝对优势。下一章将论述 DTorch 的 DTensor + Single-Controller API 如何成为 GPU 集群操作系统的标准接口，支撑起易用、低成本、高性能和高可靠性的 GPU 集群。

## 6 迈向 GPU 集群操作系统

当前 GPU 集群以容器（Pod）为最小调度单位：用户向 Kubernetes 等容器编排系统申请指定数量、指定型号的 GPU 容器，在其中运行 PyTorch 程序。对编排系统而言，容器中的 PyTorch 程序是一个黑盒——它只能看到显存占用、GPU 利用率等硬件指标，看不见里面在运行什么计算。GPU 集群由于不具备高效冻结和迁移 GPU 容器的能力（[§6.6](#66-现有的容器迁移方案)），因此无法高效调度。这种方式存在以下缺陷：

- **资源利用率低**：编排系统看不见容器里运行的计算，只能把 GPU 按整卡、整容器分配出去；分配之后，无论这张 GPU 的实际利用率多低，集群都无法收回再调度。

- **无法容错与弹性调度**：PyTorch 的 ProcessGroup 在启动时固定，发生故障时无法把任务迁移到健康节点，运行时也无法动态增删计算节点；扩缩容与故障恢复，都需要重启整个任务。

- **PyTorch API 暴露过多硬件细节**：集群将硬件的控制权整体交给 PyTorch 程序，PyTorch 又把 CUDA Graph、Stream、Event 等硬件细节直接暴露给上层代码，框架和模型的实现因此与底层硬件深度绑定；接入新芯片时，上层代码要逐一重新适配并进行精度验证——“硬件可替换、新芯片即插即用”无从谈起。

在当前的 PyTorch 生态下，GPU 集群的管理和调度大量依赖人工编排，新硬件的接入也需要用户修改程序、验证并适配。行业迫切需要一套能感知 PyTorch 计算内容、自动调度并管理整个 GPU 集群的操作系统。

DTorch 采用 Client、Controller 和 Worker 解耦的调度方案, 它们彼此之间通过消息队列通讯，如 [图 2](#image-2) 所示。这一架构正好满足 GPU 集群操作系统的需求，如 [图 1](#image-1)：用户在 Python Client 上描述的计算逻辑会被序列化为 Operator 消息流并发送给 Controller，由 Controller 完成调度后交由 GPU 集群上的 Worker 执行：

- **Client 运行在用户侧，是 GPU 集群操作系统的用户界面**，其负责构建计算节点，承载业务逻辑；
- **Controller 和 Worker 都运行在集群侧，是调度和执行单元**，Controller 持有 Client 程序请求的硬件拓扑、全局计算子图等所有信息；Worker 它只负责执行计算，是无状态的执行单元。
- 集群操作系统支持连接多个 Client，同时服务于多个用户。每个 Client 都会有一个 Controller；
- 由于 “Operator 消息流” 和 “计算结果返回” 的通信量不大，且通信可与计算相互 overlap，因此对网络的要求不高，可以在普通以太网环境下使用。

**用户通过 DTorch API 访问 GPU 集群，集群与 Controller 紧密配合，使得集群具备驱逐、冻结和迁移 Worker 的能力，可以完成弹性调度、容错等功能，实现易用、低成本、高性能和高可靠性的 GPU 集群。**

<figure markdown id="image-2">
  ![Single-Client Single-Controller Multi-Worker 架构](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/client_controller_worker_cn.png)
  <figcaption>图 2：Client、Controller 和 Worker 并发执行，并通过 Queue 异步通信</figcaption>
</figure>

### 6.1 提高资源利用率

当前 GPU 集群平均利用率普遍低于 50%<sup><a href="#note-1">注释1</a></sup>。基于 DTorch API 构建的 GPU 集群操作系统具备驱逐、冻结和迁移 Worker 的能力，通过更好的调度提高资源利用率。其调度策略也很直观：将任务按实时性和优先级区分，优先保障高优先级的实时任务；空闲资源则由低优先级、非实时任务铺满（以更低的价格引导用户将训练和推理任务定义为非实时任务，空闲时自动调度执行）。

<a id="note-1"></a>

!!! note "注释1：“当前 GPU 集群平均利用率普遍低于 50%” 的统计口径与数据来源"
    - GPU 集群平均利用率的统计口径是 nvidia-smi 显示的 GPU Utilization 在一段时间（比如一个月）内的时间平均值。
    - “50%”这个数据是个人与其他同行交流得到的经验值，暂未找到权威的统计数据。
    - [Vexxhost](https://vexxhost.com/blog/gpu-utilization-ai-infrastructure/)（云服务商）称，多数 AI 集群的 GPU 利用率仅为 30–50%，文中未说明数据来源与统计口径；
    - [DevZero](https://www.devzero.io/blog/why-your-gpu-cluster-is-idle)（GPU 开发环境厂商）文章标题即为《Why Your GPU Cluster Is 80% Idle》——集群平均利用率约 20%，并将成因归于开发类任务的“占而不跑”：研究人员预留 GPU 一周，实际使用的时间往往只有 10–15%，文中同样未说明统计口径；
    - [Run:ai](https://mlops.community/blog/this-open-source-tool-measures-gpu-cluster-utilization-heres-why-that-matters#:~:text=However%2C%20%28based%20on%20experience%20with%20POCs%20and,fully%20utilizing%20their%20GPU%20and%20AI%20hardware.) 的数据约为 14%，是三者中唯一有实测依据的：Run:ai 在一年间访谈了数十家企业，研究人员自估的集群利用率平均高达 62%；而 Run:ai 对其客户集群（POC 与新部署）的实测平均利用率仅约 14%。

    - 这个数据也和集群运行的任务、模型规模、团队的类型密切相关。Jupyter Notebook 交互式开发，以及小模型的训练推理任务，都会拉低集群的平均利用率。专门用于 LLM 训练且经过精心优化的集群，利用率可以达到 90% 以上。工程能力强的团队利用率高，小团队利用率低。

GPU 集群平均利用率低的原因有很多，需要逐一分析、针对性优化；具体成因与集群操作系统的应对策略如[表 7](#table-7)所示。这些策略本身并非高深的技术，但在现有的“基于容器（Pod）的调度方案”下，集群看不见也无法修改用户代码，优化无从落地。

<figure markdown id="table-7">
  |原因|详情|应对策略|
  |-|-|-|
  |预留与“占而不跑”|按配额给各团队预留资源；部分成员为抢占资源过量申请，却不使用|统一资源池 + 配额 + 超售：资源不足时迁移低优先级任务；迁移合并低利用率任务，回收资源|
  |开发调试|开发、调试时 GPU 间歇使用，拉低平均利用率|多人共享 GPU，显存溢出时自动迁移任务；单卡模拟分布式，调试不占多卡|
  |负载非算力密集|推理、小模型、Embedding 等任务吃显存不吃计算，GPU 利用率天然低|混合部署：把吃显存与吃计算的任务调度到同一张卡上，互补填充|
  |推理流量波动|在线推理随用户流量起伏，夜间、周末等低谷期 GPU 利用率大幅下探|弹性伸缩：低谷期回收 GPU，分配给离线训练和推理等低优先级任务|
  |数据与 I/O 瓶颈|模型加载、数据加载、数据预处理和 checkpoint 写入|常用模型预加载到 GPU 显存或内存；框架侧数据预加载与多线程预处理；异步 checkpoint；OS 收集统计信息，提示用户优化|
  |整卡独占|K8s device-plugin 默认硬隔离独占：一张卡只给一个任务，任务用不满时，空闲的算力与显存既不能共享，也无法合并|细粒度共享 + 全局调度：MIG / 时间片 / 算子级调度让多任务共用一张卡|
  |碎片化|空闲 GPU 零散分布在各节点，单节点数量不足，多机多卡作业无法启动|拓扑感知调度 + 碎片整理：按通信需求放置任务；迁移低优先级任务，归拢空闲 GPU|
  |环境初始化|镜像拉取、容器启动、torchrun 启动、NCCL 初始化等耗时数分钟|Worker 常驻成池：新作业秒级接管，免去容器和进程的创建开销|
  |故障空转|单节点故障导致整个作业重启|心跳检测秒级发现故障，任务迁移到健康节点，只重算丢失的部分|
  |框架优化不到位|框架层实现低效；调用未优化的算子|提供深度优化的框架与算子实现；OS 收集算子级统计信息，反馈给用户并做针对性优化|
  |型号绑定|用户程序绑定特定型号 GPU，其余型号闲置|统一调度：按任务需求自动匹配型号，弱化型号绑定|

  <figcaption>表 7：GPU 利用率低的原因与集群操作系统的应对策略</figcaption>
</figure>

### 6.2 容错与弹性调度

PyTorch SPMD 的进程组在启动时固定：任一 rank 故障，整个作业随之重启；扩缩容同样以重启为代价。集群操作系统则把故障处理与资源的弹性调度变成普通的调度操作。

- **故障感知**：GPU 集群操作系统可以感知计算内容、访问底层硬件，因此比只能看到容器指标的编排系统更早发现硬件故障。
- **故障恢复**：当故障发生故障时，Worker 状态分为可恢复和不可恢复两种。如果是前者，则迁移到健康的 Worker 上继续运行，上层程序对故障完全没有感知；如果是后者，则向用户 Client 抛出异常，Client 接收到异常后，从模型文件或上一个 checkpoint 恢复模型，继续执行后续计算。
- **弹性调度**：借助 DTorch API 层的 scalable 接口，任务可以动态发送到多个 Worker 上并行计算；集群操作系统具备驱逐、冻结和迁移任务的能力，能根据当前负载与队列，弹性地执行剩余任务。

### 6.3 接入新芯片：用户无感

在 PyTorch 生态下，使用新的硬件之所以困难，是因为硬件细节与适配责任被层层推给了用户：

- **API 层**：PyTorch API 直接暴露 CUDA Graph、Stream、NCCL 后端、CUDA 环境变量等大量硬件细节；芯片厂商适配 PyTorch 时，为了性能还会提供更多私有接口，上层代码因此与具体硬件深度绑定。
- **框架层**：Megatron-LM、vLLM 等框架在 NVIDIA GPU 上开发和验证，并提供一系列仅针对 NVIDIA GPU 的高性能算子，有的团队还会在此基础上维护自己的修改版本。使用其他硬件时，需要切换到芯片厂商定制的版本；它与原版、团队自己的修改版之间都存在大量差异，验证、适配与精度对齐的工作量全部落在用户身上。
- **部署层**：用户以容器方式启动程序，硬件驱动、芯片运行所需的软件栈、第三方依赖的适配与安装，需要自行逐项完成并验证。硬件故障与问题排查也需要用户自行负责。

DTorch 则基于标准 API 完全屏蔽硬件细节，新硬件的适配工作整体交由集群操作系统接管。用户希望改用其他硬件时，只需通知集群即可，上层软件无需任何改动；软件栈适配、精度验证等工作，全部由平台承担。**新芯片即插即用，用户无感。**这种模式让用户、集群平台和芯片厂商同时受益：

- **用户**：在不同硬件间无缝切换，降低使用成本。
- **集群平台**：可以将用户流量统一切换到不同硬件上，降低运营风险与成本，吸引更多用户。
- **芯片厂商**：芯片落地有了统一出口——只需完成与集群操作系统的一次适配，无需再逐一适配各训练和推理框架，也无需帮助用户逐个迁移任务。

### 6.4 优秀的用户体验

基于 DTorch 的 GPU 集群操作系统具有易用、低成本、高性能和高可靠性的特性：

- **易用**：DTorch 提供与 PyTorch 一致的 single-device API，分布式编程与单卡编程一样简单。无需购置昂贵的设备、无需配置繁琐的 GPU 环境，在任何联网设备上安装 DTorch 库、配置 API Key，即可调用远端庞大的 GPU 算力，即使是一台廉价的树莓派也是如此。
- **低成本**：按用户真实的调用量计费；统一调度摊薄单位算力成本；用户可以在不同硬件间无缝切换，选用价格更低的算力。
- **高性能**：集群对存储、网络与计算统一调度，显著提高 GPU 利用率；并提供深度优化的框架与算子实现，提高用户程序的运行速度；操作系统还可以收集程序的运行信息，提示用户可以使用哪些优化手段。
- **高可靠性**：高效的故障感知和故障恢复能力，为用户提供高可用的 GPU 服务。

### 6.5 全新的商业模式：卖硬件变成卖计算

2026 年，全球 AI/GPU 服务器硬件的市场规模已达数千亿美元，GPU 集群是 AGI 时代最重要的基础设施。对 GPU 集群平台而言，基于 DTorch API 构建的集群操作系统意味着全新的商业模式：**从卖硬件转向卖计算**。

- **卖硬件**：GPU 机器按容器（Pod）分配，容器中运行的程序对平台是黑盒；
- **卖计算**：根据用户请求的计算序列分配计算资源、返回最终的计算结果。

前者，容器一经分配，平台就必须保证资源归其独占，即使空闲也不能回收；后者，平台保留了资源的调度权，只需向用户交付正确的计算结果，因此可以在整个集群内自由调度，提高资源利用率。

在“卖计算”的模式下，用户获得易用、低成本、高性能和高可靠性的服务；平台出售计算服务、赚取合理的利润，并有充足的动力降低运营成本：更高效的调度、适配成本更低的硬件、更优的训练与推理框架和算子实现。芯片厂商则获得统一的落地出口——一次适配即可，无需再逐一适配各训练和推理框架。由此形成用户、平台与芯片厂商三方共赢的商业模式。

### 6.6 现有的容器迁移方案

现有的任务迁移方案分为两类：容器销毁并重建、容器热迁移。前者是当前的主流方案，以任务中断为代价；后者远未成熟——两者都无法为集群提供高效的容器迁移能力。

**1. 容器销毁并重建**

集群编排系统通知用户程序保存 checkpoint（前提是用户已实现该逻辑），随后销毁容器、在新机器上重启任务。整个过程无法做到对用户无感知，存在明显的缺陷：

1. 需要用户主动适配 checkpoint 的保存逻辑；
2. checkpoint 保存与任务重启非常耗时（拉取镜像、配置运行环境、读取模型文件，通常需要十分钟左右），期间机器持续闲置；
3. 若用户未实现 checkpoint 保存，迁移还会造成部分计算内容丢失。

实践中，集群平台极少主动销毁用户容器——这会强行中断用户任务、给平台自己带来麻烦，平台通常只在发现 GPU 利用率低时发送告警邮件，而大部分用户并不会理会。

**2. 容器热迁移**

热迁移方案还处于“早期商用 / 工业试点”阶段，目前最成熟的是 [CRIU](https://github.com/checkpoint-restore/criu) + [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)。该方案存在以下缺陷：

- **迁移耗时长**：需要迁移显存、CUDA 进程状态、容器内存与状态，耗时随显存占用线性增长——[DevZero 实测](https://www.devzero.io/blog/gpu-container-checkpoint-restore)，仅生成 cuda checkpoint 一项，A100 单卡约需 13 秒、4 卡约 55 秒；checkpoint 文件的大小与进程的显存占用相当，跨机传输与恢复还需额外的时间；
- **只能在软硬件完全相同的环境间迁移**：CPU 与 GPU 的型号、驱动版本、CUDA 版本等任一不同，都会导致迁移失败——CUDA 官方文档要求恢复端 GPU 与原 GPU 为同一芯片型号且显存足够（[CUDA Checkpointing Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html)），NCCL 与 CUDA Checkpoint 的配合还存在严格的库版本限制（[NCCL Roadmap](https://github.com/NVIDIA/nccl/issues/2272)）；
- **暂不支持 NCCL**：多机分布式训练依赖的 NCCL 不在支持范围内（[DevZero](https://www.devzero.io/blog/gpu-container-checkpoint-restore#current-limitations-and-requirements)），NCCL 和 gloo 通信组等均需在用户程序中适配重建，多机迁移还需要额外的协调——NCCL 官方也将「CUDA Checkpoint 支持改进（移除 deviceAPI、CUDA Graph 与严格的库版本限制）」列为尚未启动的待开发事项（[NCCL Roadmap](https://github.com/NVIDIA/nccl/issues/2272)）。

**3. DTorch 的 Worker 迁移**

与上述两种方案相比，GPU 集群操作系统的 Worker 迁移则非常高效。Worker 是无状态的执行单元，迁移时只需把计算所需的 Tensor 从一块 GPU 拷贝到另一块 GPU，且可以在任意硬件间进行；Controller 自动重建 NCCL 与 gloo 通信组，用户程序对此完全无感知。

- **推理**：模型权重是 frozen 的，可提前从模型文件加载，仅需拷贝 Activation Tensor；
- **训练**：需拷贝权重与 Adam 优化器状态等——拷贝时机可选在权重更新完成之后，避免拷贝梯度和 Activation Tensor；fp32 主权重与 fp16 副本并存时，只需拷贝 fp32 权重，fp16 副本可由其重新转换得到；
- PyTorch 内存池中预留的内存/显存与 CPU 内存无需拷贝；

综合来看，**容器迁移并没有解决 GPU 利用率低的问题，只是把问题从一台机器转移到另一台机器**——[表 7](#table-7) 所列的利用率低的问题，单靠容器迁移一个都解决不了。最关键的是，任务迁移需要与“感知容器内的计算”配合执行，并能在不同型号的 GPU 间迁移、把多个任务合并到同一块 GPU 上运行，这样才能整理显存与算力的碎片，提高集群的整体利用率。任务的算力与显存需求随时间波动，集群若无法感知容器内的计算，便只能依据利用率合并任务，算力需求回升时任务相互争用、时延升高，显存需求一旦超出单卡容量，任务便会 OOM 失败。

## 7 基于易用性，重构 PyTorch 生态

DTorch 基于 DTensor 和 Single-Controller 构建了易用的分布式 API。凭借极致的易用性，基于 DTorch API 搭建的分布式训练和推理框架可以吸引一部分用户。这套 API 同时也是 GPU 集群操作系统的标准接口——借助集群提供的易用、低成本、高性能和高可靠性的算力服务，DTorch 可以吸引大部分用户。

### 7.1 重构的机会

目前主流的深度学习分布式训练和推理框架基本都基于 PyTorch 开发，生态看似已经收敛到 PyTorch，实则还存在机会：

- **分布式易用性不足**：PyTorch 的分布式能力建立在 Multi-Controller(SPMD)之上，TP / PP / EP 等并行方式要求改写模型、手动切分权重、编排通信，分布式开发依赖资深的框架工程师；官方主推的 DTensor 开发多年，至今仍处于 alpha state and under development，未被主流的训练和推理框架广泛采用。
- **集群利用率低、可靠性差**：容器中的 PyTorch 程序对调度系统是黑盒，GPU 只能整卡分配，空闲算力无法回收再调度；进程组启动时固定，任一 rank 故障，整个作业随之重启（[6.1 节](#61-提高资源利用率)、[6.2 节](#62-容错与弹性调度)）。
- **接入新硬件成本高**：PyTorch API 暴露 CUDA Graph、Stream 等硬件细节，Megatron-LM、vLLM 等框架又与 NVIDIA GPU 深度绑定；新硬件需要构建完整的分布式软件栈，并逐个适配 PyTorch 及其上层框架，周期长且各家重复投入（[6.3 节](#63-接入新芯片用户无感)）。

**DTorch 的核心优势有两个：极致的易用性，以及提高 GPU 集群利用率的能力。**如 [6.5 节](#65-全新的商业模式卖硬件变成卖计算)所述，全球每年 AI/GPU 服务器硬件的市场规模已达数千亿美元。集群利用率只要提高 1%，节省的成本就远超重构生态所需的全部投入。极致的易用性是 DTorch 吸引用户的入口，提高集群利用率带来的成本优势则是 DTorch 的杀手锏。

**当前 PyTorch 的分布式 API 尚未完善，其分布式生态繁琐难用，这为 DTorch 留出了难得的窗口期。**PyTorch 本身是计算 API 而非解决方案：transformers、diffusers、trl、Megatron-LM、vLLM、SGLang、veRL 等解决方案框架都生长于其上；用户真正需要的是这些能直接跑模型的框架，计算 API 只是承载它们的底座。基于 DTorch API 构建同样功能的框架，就能凭借极致的易用性吸引大量用户——当年 PyTorch 正是凭借易用性战胜 TensorFlow，如今易用性的天平将倒向 DTorch。未来还会有新的模型在不同的领域不断涌出，当 DTorch 和 PyTorch 站在同样的起点时，易用性将会成为决定性的因素。

GPU 价格高昂，一台 8 卡 H800 服务器已达 200 万元人民币，高校、实验室和小公司大多无力自建 GPU 集群；即便租用，绝大多数团队也没有资深的 AI Infra 团队能把昂贵的 GPU 高效利用起来。基于 DTorch 的 GPU 集群操作系统提供了另一种选择：无需持有 GPU，按需使用即可——一个 API 就能调用原本只有顶尖团队才拥有的集群，与 GPU 相关的繁琐工作全部由集群承担。**更重要的是，用户只需为真正执行的计算付费，无需为空闲的算力付费。**昂贵的 GPU 与资深的 AI Infra 优化不再是顶尖团队的专属，而成为人人可用的共享资源——高效的算力触手可及，这将极大激发科研工作者的创新潜力。

### 7.2 实现路径

生态重构按先易后难的方式推进，分为三步：推理、训练与 GPU 集群操作系统。前期几个人即可开始，随后逐渐建立影响力、逐步加大投入。

**推理**

- Diffusion 推理（已完成示范）：diffusers 增加 DTensor 支持，对齐 SGLangDiffusion 的功能。DTorch 已基于 `diffusers==0.34.0` 迁移 SD3 / FLUX 作为示范；
- LLM 推理：transformers 增加 DTensor 支持（改动与 diffusers 示范一致），并补齐 PagedAttention、continuous batching 等推理服务优化，性能达到 vLLM 的水平。

**训练**

- 训练组件：需要支持 autograd、优化器以及 DP / PP / EP / ZeRO 等并行范式（如训练中的计算与通信重叠），完成后 transformers / trl 的分布式训练随之可用；
- RL 训练：训练与推理是同一执行引擎的两种使用方式，多角色编排由 Single-Controller 原生完成。

**GPU 集群操作系统**

核心功能包括：

- 资源池化：集群的算力、显存、网络与存储构成统一资源池，按任务与算子粒度分配、回收与共享；
- 全局调度器：拓扑感知放置、优先级与抢占、负载均衡，驱逐、冻结、迁移任务，动态扩缩容；
- 硬件抽象层：屏蔽 GPU 型号与厂商差异，新芯片只需适配这一层；
- 故障管理：心跳检测秒级发现故障，任务迁移到健康节点，只重算丢失的部分；
- 多租户隔离：任务间的算力、显存与数据隔离，配额与权限管理；
- 可观测性与计费：收集算子级运行统计，向用户提示优化空间，按真实使用量计量。

### 7.3 Why Not PyTorch

PyTorch 难以复刻 DTorch 的两个核心优势：

**1. 分布式接口的易用性**

PyTorch 分布式接口的核心是 Multi-Controller + SPMD 范式，这一范式与易用性天然冲突。[5.2 节](#52-最易用的分布式-api)已论证为何基于 Single-Controller 才能构建易用的分布式接口，此处不再赘述。

**2. GPU 集群操作系统的调度**

- PyTorch 的 Python 代码与 CUDA 执行必须位于同一进程，不支持将 CUDA kernel 的运行放到远端机器上，补齐这一能力的改造成本很高。而且调度与执行分离会带来更高的时延，这一时延问题需要系统性的解决方案。
- SPMD 范式要求多个进程，且每个 GPU 对应一个进程：例如运行 128 GPU 的分布式训练就需要 128 个进程，这些进程若运行在用户侧，会使本地机器不堪重负；若运行在集群侧，用户启动、调试任务都不方便。此外，128 个进程需同时与集群通信并同步执行，会带来巨大的通信开销与时延。

PyTorch 当初选择 SPMD，看中的正是调度不出本机：Controller 与 GPU 同机，调度只经过 PCIe 总线，没有跨网络的开销。而当 Python 程序与 CUDA kernel 分处不同的机器时，调度注定要跨越网络——SPMD 赖以成立的前提已然消失，此时再保留 SPMD，理由何在？

### 7.4 多方共赢的新生态

重构完成后，一套易用的分布式 API 将贯穿整个技术栈：向上支撑训练与推理框架，向下作为 GPU 集群操作系统的标准接口，使集群对外提供易用、低成本、高性能与高可靠的算力服务。技术栈的分工重归清晰，生态中的每一方都将从中受益：

**1. 用户**

- 通过“和 single-device 一样易用的分布式 API”调用 GPU 集群，无需编排进程与通信，开发和调试回到单卡时代。分布式编程不再需要依赖资深的框架开发工程师，算法工程师可以专注于算法与模型的优化。
- 在任何联网设备上都能访问 GPU，无需配置镜像环境，启动容器，使用方式和本地计算没有任何差别。无需自建 GPU 集群，模型、数据都托管在云端。只为真正执行的计算付费，不需要为空闲的算力付费。GPU 集群会自动优化，提高程序的运行速度，也会提供优化建议，帮助用户提高效率。
- 用户可以随时更换硬件：程序与框架无需任何改动，软件栈适配与精度验证全部由平台承担。

**2. 解决方案开发者**

- 基于 DTorch API 构建训练和推理框架，只需专注于 pipeline 与算子优化——Tensor 切分、通信与调度等分布式细节全部由 API 层承担。
- 各类优化与模型实现解耦，可自由组合、跨框架复用；RL 训练中，训练与推理共享同一份模型实现，权重同步无需格式转换，训推数值天然一致。

**3. GPU 集群平台**

- 平台对外提供易用、低成本、高性能与高可靠的算力服务，商业模式从“卖硬件”转向“卖计算”——不再按容器出租独占的 GPU，而是按计算请求分配资源、交付计算结果。
- GPU 的调度权保留在平台手中，驱逐、冻结、迁移任务与故障恢复都成为普通的调度操作；集群统一调度存储、计算和网络资源，同样的硬件承载更多计算，利用率持续提高。集群看得见计算内容，平台还能在框架层与算子层做深度优化，提高用户程序的运行速度。
- 平台掌握 API 的定义权，上层框架与用户程序都经由它访问算力。平台因此不被任何芯片厂商绑定，可以自由选用、适配性价比更优的芯片——服务更好，成本更低。

**4. 芯片厂商**

- 芯片落地有了统一的出口——只需适配集群操作系统的硬件抽象层，无需再逐一适配 PyTorch 及其上层框架，也无需协助用户逐个迁移任务。
- 新芯片即插即用，竞争回归芯片本身的性价比。

### 7.5 劣势与挑战

重构方案也面临以下挑战：

- **DTorch 的训练和推理能力尚不成熟**：目前已落地的只有扩散模型推理示范，[7.2 节](#72-实现路径)中推理与训练两步的其余工作——LLM 推理、训练组件、RL 训练——都仍在路线图中，对齐成熟训练和推理框架的性能需要投入大量的开发人力；
- **生态惯性**：现有框架的用户与维护者需要付出迁移成本，Megatron-LM、vLLM、SGLang 和 veRL 等框架的成熟度与社区积累不会一夜蒸发；
- **GPU 集群操作系统尚待开发**：[7.2 节](#72-实现路径)所列的调度、存储、通信、容错、多租户等功能均需从零构建，需要持续的人力投入与一定的开发周期；

DTorch 先以分布式接口的易用性吸引一部分用户；GPU 集群操作系统建成后，再以易用、低成本、高性能与高可靠的算力服务吸引大部分用户。分布式 API 是集群操作系统的标准接口，集群操作系统为 API 提供算力底座，二者优势互补，最终形成完整的生态。

## 8 总结

DTorch 的两个核心优势——极致的易用性与提高 GPU 集群利用率的能力——同出一源：用户描述计算逻辑，框架持有计算节点的全局描述。有了全局描述后，所有繁琐的工作都由框架承担。用户不再需要编排进程与通信，分布式编程回到 single-device 时代。集群的调度系统看得见计算内容，驱逐、冻结、迁移任务与故障恢复都成为普通的调度操作，新硬件只需适配统一的抽象层。

全球每年 AI/GPU 服务器硬件的市场规模已达数千亿美元，行业急需能感知计算内容、统一调度集群资源的 GPU 集群操作系统，谁先建成，谁就掌握下一代生态的底座。十年前，PyTorch 凭易用性从 TensorFlow 手中接过生态；分布式时代，机会将属于易用、低成本、高性能与高可靠的 GPU 集群操作系统。

## 延伸阅读

- [Github 页面](https://github.com/tingkuanpei/dtorch) — 项目源码库
- [《DTorch 介绍》](https://tingkuanpei.github.io/dtorch/cn/blog/introduction/) — DTorch 与 PyTorch 的全面对比、API 简介、精度与性能数据
- [《DTorch 架构设计：简洁与高效何以兼得》](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/) — 三大核心设计详解
- [《DTorch 的优势与机遇》](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/) — 优势、行业机会、劣势与路线图
