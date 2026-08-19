# DTorch：更易用的 PyTorch 分布式推理 API——基于 Single-Controller 与 Distributed Tensor

## 1 概述

**DTorch 是一套易用的 PyTorch 分布式推理 API**，无需多进程、无需 SPMD、无需配置 `ProcessGroup`—— 只需照常编写单卡代码，再稍作改动：将 `Tensor` 替换为 `DTensor`（携带“分布在哪些设备上、如何切分”信息的 Tensor） —— DTorch 会自动完成分布式系统的资源管理、调度和通信。

而在 PyTorch 中，多卡程序的开发成本远高于单卡程序。使用 PyTorch 编写多卡程序时，用户需要通过 torchrun 启动多个进程、遵循 SPMD（Single Program Multi Data）范式、手动管理 ProcessGroup，并从单张 GPU 的视角描述计算。当不同 GPU 需要执行不同的代码时，用户还必须在代码中以 if-else 区分各张 GPU 的行为。两者的整体对比如表 1 所示：

<figure>
  <table>
    <thead>
      <tr>
        <th colspan="2"></th>
        <th>PyTorch</th>
        <th>DTorch</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td colspan="2">编程范式</td>
        <td>Multi-Controller + SPMD</td>
        <td>Single-Controller</td>
      </tr>
      <tr>
        <td colspan="2">对 DTensor 的支持</td>
        <td><a href="https://docs.pytorch.org/docs/stable/distributed.tensor.html">alpha state</a></td>
        <td><span style="color: #4caf50; font-weight: bold;">原生支持</span></td>
      </tr>
      <tr>
        <td rowspan="4" style="text-align: center; vertical-align: middle;">分布式接口易用性</td>
        <td>DDP</td>
        <td>⭐️⭐️⭐️</td>
        <td>⭐️⭐️⭐️</td>
      </tr>
      <tr>
        <td>TP、PP&amp;EP</td>
        <td>⭐️⭐️</td>
        <td><span style="color: #4caf50; font-weight: bold;">⭐️⭐️⭐️</span></td>
      </tr>
      <tr>
        <td>强化学习</td>
        <td>⭐️</td>
        <td><span style="color: #4caf50; font-weight: bold;">⭐️⭐️⭐️</span></td>
      </tr>
      <tr>
        <td>总结</td>
        <td>差</td>
        <td><span style="color: #4caf50; font-weight: bold;">优</span></td>
      </tr>
      <tr>
        <td colspan="2">分布式接口灵活性</td>
        <td><span style="color: #4caf50; font-weight: bold;">极高</span></td>
        <td>高</td>
      </tr>
      <tr>
        <td colspan="2">编程视角</td>
        <td>分布式集群的局部视角</td>
        <td><span style="color: #4caf50; font-weight: bold;">分布式集群的全局视角</span></td>
      </tr>
      <tr>
        <td colspan="2">调度开销</td>
        <td><span style="color: #4caf50; font-weight: bold;">极低</span></td>
        <td>低</td>
      </tr>
      <tr>
        <td colspan="2">资源管理</td>
        <td>用户手动管理</td>
        <td><span style="color: #4caf50; font-weight: bold;">框架自动管理</span></td>
      </tr>
      <tr>
        <td colspan="2">任务切分</td>
        <td>用户手动切分</td>
        <td><span style="color: #4caf50; font-weight: bold;">框架自动切分</span></td>
      </tr>
    </tbody>
  </table>
  <figcaption>表 1：PyTorch 与 DTorch 分布式接口对比</figcaption>
</figure>

## 2 DTorch 和 PyTorch 示例代码的对比

下面以“创建并打印**在第 0 维被切分的分布式 Tensor**”为例，对比 DTorch 和 PyTorch 的写法：

<table>
<thead>
<tr>
<th>DTorch（单线程）</th>
<th>PyTorch（多进程）</th>
</tr>
</thead>
<tbody>
<tr>
<td valign="top">

```python
# 运行命令：python3 test.py

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
# PyTorch 需要使用 torchrun 命令创建多个进程，所有进程都执行以下代码。
# 运行命令：torchrun --standalone --nnodes=1 --nproc-per-node=2 test.py

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

两侧代码按环节对比如下：

1. **运行方式**：DTorch 以 `python3 test.py` 单进程单线程运行；PyTorch 需以 `torchrun` 启动 2 个进程，所有进程执行同一份代码（SPMD）。
2. **设备管理**：DTorch 通过 `DeviceMesh("cpu", [0, 1])` 声明设备网格，设备由框架管理；PyTorch 需手动 `init_process_group` 初始化 ProcessGroup、`set_device(rank)` 绑定 GPU，并依赖 `rank` 区分各进程的行为。
3. **数据切分**：DTorch 通过 `[Shard(0)]` 声明在第 0 维切分，切分与通信由框架自动完成；PyTorch 需以 `shape[0] // world_size` 手动计算局部 shape，每个进程仅持有 `(2, 3)` 的局部数据。
4. **数据打印**：DTorch 直接打印即输出全局 shape（`(4, 3)`）及 `device_mesh`、`placements`；PyTorch 各进程只能打印局部数据，若需完整 Tensor，须手动分配缓冲、`all_gather` 聚合并 `concat` 拼接。
5. **资源清理**：DTorch 由框架自动管理；PyTorch 需 `destroy_process_group()` 手动销毁。

归结起来：PyTorch 中“Tensor 如何分布”并未记录在程序中，由用户通过 `rank`、shape 计算与集合通信显式表达；DTorch 中该信息由 DTensor（DeviceMesh 与 Placements）携带，切分、通信与设备绑定均由框架自动完成。


## 3 背景与基础

### 3.1 Single-Controller 与 Multi-Controller

将深度学习程序从单卡扩展到多卡、多机时，系统必须回答一个根本问题：**谁来决定哪张 GPU 执行哪部分计算？** 承担 Tensor 切分、任务分发、通讯协调与资源管理的角色就是 Controller（控制者）。按 Controller 的数量与位置，分布式系统分为两种范式：

- **Multi-Controller**（PyTorch 采用）：每张 GPU 各由一个 Controller 进程管理，进程既是控制者又是执行者，直接调度本机 GPU，与其他进程通过集合通信协调；所有进程执行同一份代码，即 SPMD 范式。用户在分布式集群的局部视角下编程。代表系统：PyTorch `torch.distributed`（DDP / FSDP）、Megatron-LM、DeepSpeed。
- **Single-Controller**（DTorch 采用）：整个集群只有一个 Controller，统一管理全部 GPU 资源。用户在一个普通进程中、以分布式集群的全局视角描述计算，数据切分、任务分发与通讯协调均由框架自动完成。该范式最早在 TensorFlow v1 中被使用，并在 [Pathways](https://arxiv.org/abs/2203.12533) 中有详细论述。代表系统：TensorFlow v1、Pathways、DTorch。

两者的架构对比如图 1 所示；概念的完整介绍见 [Single-Controller 与 Multi-Controller](https://tingkuanpei.github.io/dtorch/cn/developer_guide/single_and_multi_controller/)。

<figure markdown>
  ![Single-Controller 与 Multi-Controller 对比](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/single_controller_vs_multi_controller.png)
  <figcaption>图 1：Single-Controller 与 Multi-Controller 架构对比</figcaption>
</figure>

Single-Controller 和 Multi-Controller 是 DTorch 和 PyTorch 最根本的差异（PyTorch 中也实验性支持了 DTensor，因此 DTensor 不是两者根本性的差异）。两者对比如表 2 所示：

<figure markdown>
  ||Single-Controller|Multi-Controller|备注|
  |-|-|-|-|
  |编程视角|<span style="color: #4caf50; font-weight: bold;">分布式集群的全局视角</span>|分布式集群的局部视角||
  |分布式接口的易用性|<span style="color: #4caf50; font-weight: bold;">好</span>|差||
  |资源管理|<span style="color: #4caf50; font-weight: bold;">系统自动管理</span>|用户手动管理|全局设备管理、设备虚拟化、节点故障自动恢复|
  |任务切分|<span style="color: #4caf50; font-weight: bold;">系统自动切分</span>|用户手动切分|避免通信时发生死锁、auto parallel|
  |分布式接口的灵活性|高|<span style="color: #4caf50; font-weight: bold;">极高</span>||
  |调度开销|低|<span style="color: #4caf50; font-weight: bold;">极低</span>|DTorch 降低了 Single-Controller 的调度开销|

  <figcaption>表 2：Single-Controller 与 Multi-Controller 对比</figcaption>
</figure>

#### 3.1.1 Single-Controller 的优势

Single-Controller 的优势有：1. 分布式接口的易用性好；2. 基于分布式集群的全局视角，可以提供自动资源管理和任务划分的功能。

DTorch 选择 Single-Controller 最大的理由是易用性。Single-Controller 在分布式集群的全局视角上描述计算内容，这天然地与用户的思维习惯一致。Single-Controller + DTensor 的方案可以简洁地表示深度学习分布式计算所需的并行方案（Data Parallel、Tensor Parallel、Pipeline Parallel、MoE Parallel、ZeRO 和强化学习训练流程调度等），可以极大降低分布式代码的开发、修改、维护和调试成本。

其次，Single-Controller 提供了计算设备（CPU、GPU）和计算图的全局视野，可以实现全局设备管理、设备虚拟化、节点故障自动恢复、避免通信时发生死锁、auto parallel、JIT 编译（torch.compile）等功能。

#### 3.1.2 Single-Controller 的劣势

Single-Controller 的劣势有：1. 系统调度的开销较高；2. 接口的灵活性不如 Multi-Controller。

调度开销方面：Multi-Controller 的 Controller 与 GPU 同机，调度仅需经过 PCIe；Single-Controller 调度远端 GPU 需经过跨机网络通信，因此开销更大。为缓解这一问题，DTorch 采用 Single-Client Single-Controller Multi-Worker 异步架构，各组件间异步执行，详见博客 [《DTorch 架构设计：简洁与高效何以兼得》](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/)。

由于 Multi-Controller 中每个进程上执行的代码可以完全不一样，因此可以支持 MPMD（Multi Program Multi Data）的范式，其灵活性极高。理论上，任何需要并行编程的代码均可使用 MPMD 的范式实现。目前大模型已经收敛至 Transformer 架构，MPMD 范式过高的灵活性并未带来收益，反而其易用性差的劣势正在凸显。

### 3.2 PyTorch 分布式接口

#### 3.2.1 接口简介

PyTorch 的分布式接口采用了 SPMD 的范式，其为每张 GPU 创建一个管理进程，每个进程运行同样的代码，进程通过 rank id 进行区分，并提供 ProcessGroup 组件实现进程间的[集合通信](https://docs.pytorch.org/docs/stable/distributed.html#backends)。由于每张 GPU 都有自己的管理进程，因此将这种范式归类为 **Multi-Controller**。其示例代码如下：

```Python
# 运行命令： torchrun --standalone --nnodes=1 --nproc-per-node=2 test.py
# 运行的进程数 = nnodes * nproc-per-node

import torch
import torch.distributed as dist

def main():
    # 初始化 ProcessGroup 并获取 world_size、rank 变量
    # world_size: 进程数
    # rank: 进程编号, 范围为 [0, world_size)
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 指定当前进程绑定的 cuda device id
    torch.cuda.set_device(rank)

    # 创建 shape=(4, 3) 的 Tensor，并将其在第 0 维度切分，每张卡持有 Tensor 的一部分。
    shape_total = (4, 3)
    shape_this_rank = (shape_total[0] // world_size, shape_total[1])
    x = torch.rand(shape_this_rank, device='cuda')

    # rank=0, x.shape=torch.Size([2, 3]), cuda:0
    # rank=1, x.shape=torch.Size([2, 3]), cuda:1
    print(f"{rank=}, x: {x.shape}, {x.device}")

    # 从所有进程聚合 tensors，每张卡均持有完整的 Tensor
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, x)
    gathered_tensors = torch.concat(gathered_tensors, dim=0)

    # rank=0, gathered_tensors.shape=torch.Size([4, 3]), cuda:0
    # rank=1, gathered_tensors.shape=torch.Size([4, 3]), cuda:1
    print(f"{rank=}, gathered_tensors: {gathered_tensors.shape=}, {gathered_tensors.device}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

#### 3.2.2 演进历史

Data Parallel 是最早出现的并行训练方式：每张 GPU 持有完整的模型，各自处理不同 batch 的数据，前向、反向计算后通过 AllReduce 对梯度求平均，得到与单机训练一致的梯度。为了实现 Data Parallel，PyTorch 基于 SPMD 的范式实现了 [DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#getting-started-with-distributed-data-parallel)。SPMD 的范式和 Data Parallel 非常匹配：创建多个进程，每个进程执行一样的代码，处理不同的数据。

在 Megatron-LM 中沿用了 SPMD 的范式实现 Tensor Parallel 和 Pipeline Parallel。但此时每个进程执行的代码并非完全相同，中间需要插入条件判断，在不同进程上执行不同的代码（通信时需要获取对应的 TP 或 PP ProcessGroup；PP 中第一个 rank 和最后一个 rank 需要分别执行前后处理代码）。此时更合理的叫法应该是 MPMD（Multi Program Multi Data）。

LLM 推理和强化学习训练中，需要有中心控制节点负责与客户端交互（LLM 推理中主进程监听端口、执行引擎并返回结果）或调度计算流程（强化学习中需要执行推理、reward 和训练）。vLLM 和 veRL 为此引入了 Ray，构成了 Single-Controller + Multi-Controller 的混合范式。

受其他框架的启发，PyTorch 中也实现了 DTensor，但其依旧处于 [alpha state and under development](https://docs.pytorch.org/docs/stable/distributed.tensor.html)。PyTorch 的 DTensor 暂未在主流的大模型训练推理框架中被广泛使用。

PyTorch 分布式接口的演进历程及各阶段的适配程度如表 3 所示：

<figure markdown>
  ||时间|范式|PyTorch 适配程度|DTorch 适配程度|
  |-|-|-|-|-|
  |单卡 Eager|2016 年|Single-Controller|⭐️⭐️⭐️|⭐️⭐️⭐️|
  |DDP|2018 年|SPMD（Multi-Controller）|⭐️⭐️⭐️|⭐️⭐️⭐️|
  |TP、PP&EP|2019~2022 年|MPMD（Multi-Controller）|⭐️⭐️|⭐️⭐️⭐️|
  |强化学习|2024 年|Single-Controller + Multi-Controller|⭐️|⭐️⭐️⭐️|

  <figcaption>表 3：PyTorch 分布式接口的演进历程</figcaption>
</figure>

综上所述，PyTorch 目前的分布式接口并非经过精心设计，而是从 Data Parallel 开始沿用至今。理论上，PyTorch 的 SPMD 范式（由于每个进程的代码可以完全不同，更准确的叫法应该是 MPMD）可以实现任意的并行方式，但过高的自由度也给用户带来了较高的实现、修改和调试成本。目前 Transformer 模型单机单卡的训练和推理在 [transformers](https://github.com/huggingface/transformers)、[diffusers](https://github.com/huggingface/diffusers) 和 [trl](https://github.com/huggingface/trl) 等仓库中均有清晰且易用的实现；但当用户希望使用多卡进行训练或推理时，无需修改源码的只有 Data Parallel 和 ZeRO，其他的并行算法均需要定制化的修改（如 Megatron-LM 中 Tensor Parallel、Pipeline Parallel 和 Expert Parallel 的实现；veRL 为实现强化学习算法的并行训练提出了 HybridFlow 的抽象）。在实践中，通常由经验丰富的工程开发人员对代码进行魔改，算法人员使用魔改后的代码进行训练和推理任务，这一合作方式已经严重限制了深度学习算法的创新。过去，PyTorch 依靠易用性赢得开发者的青睐，但在分布式时代，PyTorch 却站在了易用性的对立面。

### 3.3 Distributed Tensor

DTensor(Distributed Tensor) 是描述“一个 Tensor 如何被切分并分布到多个设备上”的描述方法。DTensor 比普通 Tensor 多了两个属性： DeviceMesh 和 Placements。

- DeviceMesh：n-dimensional array，描述集群中 device 的拓扑。
- Placements：1D array of length n。描述 Tensor 在 DeviceMesh 每一维上的切分策略切分策略，切分策略共有三种：
    1. `Shard(dim)`：沿第 dim 维均匀切分，每个设备持有 1/N 的数据；
    2. `Replicate()`：完整复制到所有设备；
    3. `Partial()`：各设备持有一部分结果，求和后才是完整值。

DTensor 的示例如图 2 所示；DeviceMesh 与 Placements 的完整入门介绍见 [Distributed Tensor Overview](https://tingkuanpei.github.io/dtorch/cn/user_guide/distributed_tensor_overview/)。

<figure markdown>
  ![DTensor 概念图](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/dtensor_cn.png)
  <figcaption>图 2：DTensor 的 DeviceMesh 与 Placements 示意</figcaption>
</figure>

## 4 DTorch API 简介

DTorch 的 API 和 PyTorch 单机单卡 API 非常相似，熟悉 PyTorch 的用户可以无缝切换到 DTorch API。在 DTorch 中，用户通过单线程的 Python 代码以 Eager 模式描述计算流程。DTorch 通过 DTensor(Distributed Tensor) 实现分布式计算的支持，其原生支持 DTensor 的创建、计算和通信。

Python API 的使用指南见 [Python API Overview](https://tingkuanpei.github.io/dtorch/cn/user_guide/python_api_overview/)。

### 4.1 DTensor 的创建

DTensor 的创建 API 和 PyTorch Tensor 的创建 API 基本一致，同样支持 empty、ones、rand 等操作，调用这些接口时只需额外提供 DeviceMesh 和 Placements 参数。DTorch 中创建 DTensor 的示例代码如下：

```Python
import dtorch

shape = (4, 2)

# 1. 单卡 Tensor
# 类似 PyTorch 的构造方式
a = dtorch.rand(shape, device="cuda")

# 使用 DeviceMesh 构造
a = dtorch.rand(shape, device_mesh=dtorch.DeviceMesh("cuda", [0]))


# 2. 1D 切分 DTensor
# device_mesh 表示 Tensor 分布在 cuda:0 和 cuda:1 两张卡上。
device_mesh = dtorch.DeviceMesh("cuda", [0, 1])

# Shard(0) 表示在两张卡上按照第 0 维切分，即每张卡存储 shape 为 (2, 2) 的 Tensor。
a = dtorch.ones(shape, device_mesh=device_mesh, placements=[dtorch.Shard(0)])

# Partial() 表示在两张卡上以“部分和”的方式存储：每张卡均存储 shape 为 (4, 2) 的 Tensor，
# 但所有卡上的 Tensor 逐元素求和后才是完整的 Tensor。
a = dtorch.ones(shape, device_mesh=device_mesh, placements=[dtorch.Partial()])

# Replicate() 表示在两张卡上以“复制”的方式存储：每张卡均存储完整的 shape 为 (4, 2) 的 Tensor。
a = dtorch.ones(shape, device_mesh=device_mesh, placements=[dtorch.Replicate()])

# 可以直接打印 DTensor
# tensor([[1., 1.],
#         [1., 1.],
#         [1., 1.],
#         [1., 1.]], device='cuda:0')
# shape=torch.Size([4, 2]), dtype=torch.float32,
# device_mesh=DeviceMesh('cuda', [0, 1]), placements=[Replicate()]
print(a)


# 3. 2D 切分 DTensor
# DTensor 支持 N-D 切分，可以同时支持 Data Parallel、Tensor Parallel、
# Sequence Parallel 和 Context Parallel 等各种并行方式。
# 将 shape 为 (4, 2) 的 Tensor 切分到 2*2 的 device 上，每个 device 持有 shape 为 (2, 1) 的 Tensor。
# Placements 是长度与 device_mesh 维度数相同的 list，每个成员描述 Tensor 在对应 mesh 维度上的切分方式。
device_mesh = dtorch.DeviceMesh("cuda", [[0, 1], [2, 3]])
placements = [dtorch.Shard(0), dtorch.Shard(1)]
a = dtorch.ones(shape, device_mesh=device_mesh, placements=placements)

# tensor([[1., 1.],
#         [1., 1.],
#         [1., 1.],
#         [1., 1.]], device='cuda:0')
# shape=torch.Size([4, 2]), dtype=torch.float32,
# device_mesh=DeviceMesh('cuda', [[0, 1], [2, 3]]), placements=[Shard(dim=0), Shard(dim=1)]
print(a)


# 4. 与 PyTorch Tensor 的交互
import torch

torch_x = torch.rand((2, 2), dtype=torch.float16, device="cuda:0")
dtorch_x = dtorch.Tensor(
    torch_x, device_mesh=dtorch.DeviceMesh("cuda", [0, 1]), placements=[dtorch.Shard(1)]
)

# tensor([[0.9668, 0.0385],
#         [0.1194, 0.5146]], device='cuda:0', dtype=torch.float16)
print(dtorch_x.to_torch())
```

### 4.2 DTensor 的计算和通信

DTorch 中的所有算子均原生支持 DTensor，因此直接调用所需的算子即可。框架会自动根据输入 Tensor 的 DeviceMesh 和 Placements 执行计算，并自动推导输出 Tensor 的 DeviceMesh 和 Placements。

修改 DTensor 的 DeviceMesh 和 Placements，只需调用 Tensor.redistribute() 并指定目标 device_mesh 和 placements；也可以调用 Tensor.redistribute_by_dict()，按 mesh 维度名指定目标 Placements（目标 mesh 上不存在的维度名会被自动忽略）。用户不需要显式管理 [ProcessGroup](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)，也不需要调用 [all_reduce](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) 等集合通信算子。

```Python
import unittest

import torch

import dtorch
from dtorch.distributed_spec import init_device_mesh, Replicate, Shard
from dtorch.test.test_util import assert_tensor_allclose

class TestTensor(unittest.TestCase):
    def test_tensor(test_case):
        x_shape = [3, 4, 5]
        y_shape = [4, 1]
        dtype=torch.float16
        device="cuda:0"

        device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(2,))

        torch_x = torch.rand(*x_shape, dtype=torch.float16, device=device)
        torch_y = torch.rand(*y_shape, dtype=torch.float16, device=device)
        torch_z = torch_x + torch_y

        # dtorch.Tensor create from torch.Tensor
        dtorch_x = dtorch.Tensor(
            torch_x, device_mesh=device_mesh, placements=[Shard(1)]
        )
        dtorch_y = dtorch.Tensor(
            torch_y, device_mesh=device_mesh, placements=[Replicate()]
        )
        dtorch_z = dtorch_x + dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)
        test_case.assertTrue(dtorch_z.device_mesh == device_mesh)
        test_case.assertTrue(dtorch_z.placements == [Shard(1)])

        # tensor support redistribute
        dtorch_x = dtorch_x.redistribute(placements=[Shard(0)])
        test_case.assertTrue(dtorch_x.placements == [Shard(0)])
        dtorch_z = dtorch_x + dtorch_y
        assert_tensor_allclose(test_case, torch_z, dtorch_z)
        test_case.assertTrue(dtorch_z.device_mesh == device_mesh)
        test_case.assertTrue(dtorch_z.placements == [Shard(0)])


if __name__ == "__main__":
    unittest.main()
```

### 4.3 对并行算法的支持与示例

DTorch 中通过 DTensor 的 DeviceMesh 和 Placements 描述 Data Parallel、Tensor Parallel、Context Parallel 和 Pipeline Parallel 等并行算法。详情可参考  [Module 并行](https://tingkuanpei.github.io/dtorch/cn/user_guide/module_parallel/)：

1. Data Parallel 是 Tensor 在 batch 维度的切分，即将 Placements 设置为 Shard(0)。
2. Tensor Parallel 是将权重 Tensor 和激活 Tensor 在对应的维度 Shard（如 ColumnParallelLinear 按输出列切分、RowParallelLinear 按输入行切分），框架自动插入所需的通信。
3. Context Parallel 是将 Q/K/V 在序列维度切分。DeviceMesh 含 ulysess_cp / ring_cp 维度时，dtorch.nn.functional.scaled_dot_product_attention 会自动启用对应的 CP 实现，在算子内部完成通信，并产生对应的输出。
4. Pipeline Parallel 是不同 PP Stage 的 Tensor 使用不同的 DeviceMesh（通过 device_mesh.unbind("pp") 展开各 stage 的子 mesh），并通过 tensor.redistribute 在不同 Stage 之间传输激活。

综上，DTorch 中实现不同的并行算法时，分布式代码和单机单卡的代码一致，仅仅需要使用对应的 DeviceMesh 和 Placements 即可。以 Transformer 模型（Llama）为例的 DP、TP、PP、CP 任意组合完整实现见 [Llama 并行示例](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/) 和 [`python/dtorch/test/modules/llama.py`](https://github.com/tingkuanpei/dtorch/blob/main/python/dtorch/test/modules/llama.py)。

## 5 精度与性能对比

目前 DTorch 已经完成了原型开发：基于分布式集群的全局视角构建并实现了一套分布式 API，并基于此 API 实现了 Diffusion 模型的单机多卡分布式推理框架。

DTorch 基于 Single-Controller + DTensor 方案，其调度远端 GPU 需经过跨机网络通信，因此开销更大。为缓解这一问题，DTorch 采用 Single-Client Single-Controller Multi-Worker 异步架构，各组件间异步执行，详见博客 [《DTorch 架构设计：简洁与高效何以兼得》](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/)。本章节仅介绍 DTorch 和 PyTorch 之间的性能测试数据。

### 5.1 精度

PyTorch 算子和 DTorch 算子调用的都是 LibTorch 后端，因此其运行同样的 CUDA kernel，在输入和计算逻辑完全一致的情况下，输出可以做到逐 bit 一致（使用 [torch.equal](https://docs.pytorch.org/docs/stable/generated/torch.equal.html) 而非 [torch.allclose](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html) 进行 Tensor 的比较）。DTorch 中，所有算子的单机单卡测试均通过了一致性测试（和 PyTorch 实现对比，对输出 Tensor 使用 [torch.equal](https://docs.pytorch.org/docs/stable/generated/torch.equal.html) 判定相等）。StableDiffusion3 模型在单机单卡上也通过了相同的测试，两者的生成图像对比如图 3 所示。

<figure markdown>
  |PyTorch|DTorch|
  |-|-|
  |![PyTorch 生成的 SD3 图像](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/torch_sd3.jpg)|![DTorch 生成的 SD3 图像](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/dtorch_sd3_iter_0.jpg)|

  <figcaption>图 3：StableDiffusion3 生成图像对比（左：PyTorch，右：DTorch）</figcaption>
</figure>

### 5.2 显存占用

受益于 Client、Controller 和 Worker 异步执行的特性，DTorch 在计算过程中的中间 Tensor 可以及时释放，因此其显存占用比 PyTorch 更低。StableDiffusion3 模型的实测数据如表 4 所示：

<figure markdown>
  ||PyTorch|DTorch|
  |-|-|-|
  |峰值占用|18.131GB|17.663GB<span style="color: #4caf50; font-weight: bold;">（-2.58%）</span>|

  <figcaption>表 4：StableDiffusion3 显存峰值占用对比</figcaption>
</figure>

下面提供了一个简单的代码片段，可以解释此问题。变量 a、b、c 是计算过程的中间变量，由于没有及时释放，会增加 PyTorch 的峰值显存。而 DTorch 采用异步执行：描述计算的 Python 线程执行完 func() 函数后即返回，C++ 执行引擎由此可以得知中间变量 a、b、c 不再被持有，因此可以及时释放其显存。

```Python
def func():
    a = torch.rand(...)
    b = operator0(a)
    c = operator1(b)
    d = operator2(c)
    return d
```

### 5.3 性能

#### 5.3.1 算子层面

理论上 Single-Controller 的调度开销大于 Multi-Controller，但受益于 DTorch 在 C++ 层的深度优化，DTorch 在小算子上的开销仅比 PyTorch 高 37%；而在大算子上，两者耗时基本一致。

图 4 是不同 shape 的 Tensor 在 CPU 和 GPU 上执行加法的耗时。即使是 `Shape=(1,)` 的 Tensor 加法，DTorch 的开销也仅比 PyTorch 高 37%；而随着 Tensor 变大，两者的耗时基本一致。在计算密集的 SDPA 算子上也有相同的结论（图 5）。

<figure markdown>
  ![不同 shape 的 Tensor add 算子的耗时](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/tensor_add_performance.png)
  <figcaption>图 4：不同 shape 的 Tensor add 算子耗时（NVIDIA 4090）</figcaption>
</figure>

<figure markdown>
  ![不同 shape 的 Tensor SDPA 算子的耗时](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/sdpa_performance.png)
  <figcaption>图 5：不同 shape 的 Tensor SDPA 算子耗时（NVIDIA 4090）</figcaption>
</figure>

#### 5.3.2 模型层面

而在运行整个模型时，由于 DTorch 采用了 Client、Controller 和 Worker 异步执行的特性，DTorch 的 Python Client 执行时间比 PyTorch 少了 64%，因此有充足的时间 overlap 系统调度的通信开销。甚至由于 CUDA Kernel launch 得更及时且密集，可以小幅降低时延（当 Python 代码的运行耗时大于 CUDA Kernel 的执行耗时时，GPU 会因等待而闲置，这部分开销也被称为 “CPU 的开销”，DTorch 将其 overlap 掉了）。StableDiffusion3 单卡推理的实测数据如表 5 所示：

<figure markdown>
  |StableDiffusion3|PyTorch|DTorch|备注|
  |-|-|-|-|
  |Python Client 执行时间|0.575s|0.206s<span style="color: #4caf50; font-weight: bold;">（-64.17%）</span>|单卡推理的 CPU 耗时|
  |单卡耗时（GPU）|1.683s|1.648s<span style="color: #4caf50; font-weight: bold;">（-2.08%）</span>|单卡推理的端到端耗时|

  <figcaption>表 5：StableDiffusion3 单卡推理性能对比</figcaption>
</figure>

**Nsight System Profile**

图 6 是 Nsight System Profile 的结果。由于 DTorch 提供了完整的异步计算组件（Client、Controller 和 Worker 异步执行 + 异步获取 Tensor 的值），因此可以 overlap 掉 CPU 的开销。这避免了：1. 获取模型输出 Tensor 的开销；2. 大量小算子导致的 CPU launch kernel 的开销（text_encoder 的 input shape 很小，因此 CPU 的耗时远大于 GPU 的耗时）。

<figure markdown>
  ![Nsight System Profile 结果](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/nsys_sd3_async_get_tensor_cn.png)
  <figcaption>图 6：StableDiffusion3 单卡推理的 Nsight System Profile 结果</figcaption>
</figure>

### 5.4 PyTorch 中很难支持“异步获取 Tensor 值”

支持“异步获取 Tensor 的值”的特性有两种方案：1. 多线程；2. 协程。

**多线程**

由于 Python GIL 的存在，在 Python 中不存在真正的多线程。基于多线程实现异步，会遇到 GIL 导致的性能问题。

**协程**

可以通过协程实现异步获取 Tensor 的值，但是 PyTorch 的 API 并未原生支持协程，主要有两个障碍：

1. 诸如 [Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html) 等算子会触发 CPU 与 GPU 之间的同步等待，此时当前协程因阻塞而无法释放（即无法结束或回收），事件循环将无法切换至其他协程继续执行。

2. [CUDA 流队列中可容纳的未执行内核（kernel）数量存在上限](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-scale-launch-queues)，一旦达到该上限，线程同样会进入阻塞等待状态，直到队列中有空闲槽位为止。

而 DTorch 中通过协程 [`await TensorFuture`](https://tingkuanpei.github.io/dtorch/cn/user_guide/python_api_overview/#-await-tensorfuture) 的方式支持“异步获取 Tensor 的值”，同时由于 DTorch 采用的 Client、Controller 和 Worker 异步执行的特性，不会遇到上述的两个障碍。

## 6 展望

Single-Controller + Distributed Tensor 是一条颇具潜力的技术路线，有望重塑当前基于 PyTorch 的分布式训练和推理生态。关于 DTorch 的差异化优势与行业机遇的进一步讨论，参见博客 [《DTorch 的优势与机遇》](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/)。

## 7 总结

本文介绍了 DTorch：基于 Single-Controller 与 Distributed Tensor 的 PyTorch 分布式推理 API。

在分布式编程范式上，PyTorch 的 Multi-Controller + SPMD 要求用户以单卡视角编写多进程代码，手动管理 ProcessGroup、切分数据并调用集合通信，代码的理解、修改和调试成本较高；DTorch 则采用 Single-Controller + DTensor：用户以单线程、全局视角描述计算，只需将 Tensor 替换为 DTensor（声明 DeviceMesh 与 Placements），资源管理、任务切分与通信均由框架自动完成，多卡程序的写法与单卡程序几乎一致。

实测表明，在输出逐 bit 一致的前提下，DTorch 的显存峰值占用比 PyTorch 低 2.58%，单卡推理端到端时延低 2.08%，Python Client 执行时间低 64.17%。

DTorch 的代码和文档均在 [GitHub](https://github.com/tingkuanpei/dtorch) 开源，欢迎关注与参与。

## 8 寻求支持

DTorch 所基于的 [Single-Controller + Distributed Tensor 架构](https://tingkuanpei.github.io/dtorch/cn/blog/architecture/)是一条[颇具潜力的技术路线](https://tingkuanpei.github.io/dtorch/cn/blog/advantages_and_opportunities/)，然而仅凭个人的资源，还不足以将所有构想一一实现。如果你对这一方向感兴趣，欢迎联系 **peitingkuan@163.com**。


## 9 致谢

DTorch 在设计和实现中参考了 [PyTorch](https://github.com/pytorch/pytorch)、[OneFlow](https://arxiv.org/abs/2110.15032)、[Pathways](https://arxiv.org/abs/2203.12533)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[vLLM](https://github.com/vllm-project/vllm) 和 [veRL](https://arxiv.org/html/2409.19256v1) 等项目。

## 10 参考链接

1. [PyTorch Distributed Tensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
2. [torchtitan](https://github.com/pytorch/torchtitan)
3. [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/abs/2110.15032)
4. [Pathways: Asynchronous Distributed Dataflow for ML](https://arxiv.org/abs/2203.12533)
5. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
6. [vLLM](https://github.com/vllm-project/vllm)
7. [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/html/2409.19256v1)
8. [解读谷歌 Pathways 架构（一）：Single-controller 与 Multi-controller](https://zhuanlan.zhihu.com/p/495592456)
9. [重读 Google 旧文 Pathways，寻找 veRL 中 Single-controller 思想源头](https://zhuanlan.zhihu.com/p/1911558458903335293)
