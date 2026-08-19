# Single-Controller 与 Multi-Controller

将深度学习程序从单卡扩展到多卡、多机时，系统必须回答一个根本问题：**谁来决定哪块 GPU 执行哪部分计算？** 按控制权的放置方式，主流分布式系统分为两种范式：**Multi-Controller**（每块 GPU 一个控制者，PyTorch 采用）与 **Single-Controller**（全集群一个控制者，DTorch 采用）。本文介绍这两个概念，作为阅读开发者指南其余文档的前置背景知识。

## 1. Controller 是什么

深度学习程序由一系列 Tensor 和 Operator 组成，单卡运行时由一个进程直接驱动 GPU。扩展到多卡后，需要有角色完成：

- 把 Tensor 切分（或复制）到各设备上
- 决定每个设备执行哪些 Operator
- 在合适的时机插入 all-reduce、all-gather 等集合通讯
- 管理设备、进程、通讯组（ProcessGroup）等资源

承担这些职责的角色就是 **Controller**（控制者）。控制者的数量与位置不同，就得到了两种范式：

<figure markdown>
  ![Single-Controller vs Multi-Controller](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/single_controller_vs_multi_controller.png)
  <figcaption>图 1：Single-Controller(左图)和 Multi-Controller(右图)的示意图</figcaption>
</figure>


## 2. Multi-Controller

**Multi-Controller** 为每块 GPU 启动一个进程，每个进程既是控制者又是执行者：直接调度本机 GPU，与其他进程通过集合通讯（NCCL）协调。所有进程运行**同一份代码**、各自处理不同的数据，即 **SPMD**（Single Program Multi Data）范式。代表系统：PyTorch `torch.distributed`（DDP / FSDP）、Megatron-LM、DeepSpeed。

以 PyTorch 为例，把两个 Tensor 切到 4 张卡上相加：

```python
# 启动方式：torchrun --nproc_per_node=4 main.py
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")        # 手动创建进程组
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
rank, world_size = dist.get_rank(), dist.get_world_size()

x = torch.randn(256, 128, device="cuda")
y = torch.randn(256, 128, device="cuda")
x = x.chunk(world_size)[rank]                  # 手动切分：本 rank 只取自己的数据分片
y = y.chunk(world_size)[rank]

z = x + y                                      # 各进程独立计算自己的分片（局部视角）
```

Multi-Controller 的特征：

- **局部视角**：代码处处以 `rank` 为中心思考——"我是哪个进程？我的数据是哪一份？"
- **手动协调**：进程组创建、数据切分、集合通讯调用均由用户完成
- **调度开销极低**：每个进程只经 PCIe 总线控制本机 GPU

## 3. Single-Controller

**Single-Controller** 在整个集群中只保留一个控制者，由它统一管理全部 GPU 资源。用户在一个普通进程中以**全局视角**描述计算，数据切分、任务分发与通讯协调均由框架自动完成。该模式最早在 TensorFlow v1 中使用，[Pathways](https://arxiv.org/abs/2203.12533) 中也有论述；代表系统：TensorFlow v1、Pathways 与 DTorch。

同样的"4 卡切分、相加"用 DTorch 描述：

```python
import dtorch
from dtorch.distributed_spec import init_device_mesh, Shard

mesh = init_device_mesh("cuda", (4,), mesh_dim_names=["dp"])           # 声明集群拓扑

x = dtorch.randn(256, 128, device_mesh=mesh, placements=[Shard(0)])    # 框架自动切分
y = dtorch.randn(256, 128, device_mesh=mesh, placements=[Shard(0)])

z = x + y                             # 全局视角：一行代码描述整个集群的计算
```

没有 `torchrun`、没有 `rank` 分支、没有手动切分——Tensor 始终以完整的逻辑形态出现在代码中，"分布在哪些设备上"只是它的 Placements 属性。

Single-Controller 的特征：

- **全局视角**：与用户的思维习惯一致，分布式代码接近单卡程序
- **自动协调**：切分、通讯、资源管理由框架隐式完成
- **掌握全局状态**：控制者持有全局计算图，可进行计算-通讯重叠、算子融合等全局优化

代价是调度开销较高：控制远端机器的 GPU 需要经过跨机网络。DTorch 通过异步执行摊销这一开销。

## 4. 延伸阅读

两种范式在易用性、灵活性、调度开销等维度上的详细对比见 [Single-Controller 架构](single_controller.md)。其他阅读路径：

- [关键概念](key_concept.md) — DTorch 三大核心设计概览
- [设计决策](design_decisions.md) — 为何选择 DTensor + Single-Controller
- [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md) — DeviceMesh 与 Placements 概念入门
