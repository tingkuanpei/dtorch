# DTorch 的差异化优势与行业机遇

DTorch 是基于 Single-Controller 和 DTensor 构建的 PyTorch 分布式 API

## 1 DTensor 的优势

避免冗余的代码，提高代码的可读性和编程的难度。变得和 single-device program 一样简洁易用。提到 Megatron-LM 和 vllm 中的实现。

## 2 Single-Controller 的优势

基于 DTensor 之后，就没有必要使用 Multi-Controller + SPMD 方案， Single-Controller 还带来了分布式系统的自动通讯，调度等，可以实现故障排查，错误设备驱逐，自动恢复等功能。

放在不同 device 上进行计算的编排：RL的训练和推理的编排

## 3 实现单机和分布式计算统一的框架

## 4 实现训练和推理统一的框架
rl训推一致性对齐

## 5 接入新的算法更方便

## 6 按时间片付费的gpu
云平台只需要负责存储和计算。不需要理会业务逻辑。

## 7 基于 LibTorch，开发成本可控

易用性对于深度学习框架至关重要。回顾深度学习框架的发展历程，与其他同期框架相比，PyTorch 在性能上往往处于劣势地位，但其凭借易用性的优势吸引了大量开发者，渐渐成为事实上的行业标准。DTorch 基于 Single-Controller + DTensor 的技术路线，其在分布式的场景上，易用性优于 PyTorch。以此为突破点，可以重构当前 Transformer 模型的预训练、监督微调和强化学习训练流程，并支持未来出现的新的训练范式。在 Single-Controller 的技术路线上，存在大量的开发工作，足以支撑一个独立框架的长期发展。同时其基于 PyTorch 的 C++ 接口（LibTorch）开发，避免重复造轮子，极大地降低了框架的开发成本，只需几人的小团队，即可完成功能的开发和迭代。

## 8 行业的机会

目前流行的深度学习分布式训练和推理框架基本都基于 PyTorch 开发，看似整个生态已经收敛到 PyTorch，实则还存在一些机会：1. PyTorch 在分布式上的易用性不足，对 DTensor 的支持不够完善；2. PyTorch 的定位是框架而非解决方案，依托于 PyTorch，可以衍生解决各个问题的解决方案；3. PyTorch 对国产 GPU 的支持较差。

为了和 PyTorch 形成错位竞争，DTorch 的定位是针对具体领域的分布式计算的解决方案，如 Diffusion Transformer 模型的推理、量化训练及推理、强化学习训练等。实现解决方案并将代码开源，可以凭此证明并推广 DTorch 分布式 API，吸引其他开发者参与到 DTorch 的开发中。DTorch 把重心放在具体问题的解决方案，而非做完善的基础框架，可以小步快跑，逐步增加人力投入，并产生相应的业务价值，这也符合目前“公司对团队偏向短期考核”的现状。

## 9 展望

Single-Controller 和 Multi-Controller 是 DTorch 和 PyTorch 最根本的差异。PyTorch 的 Multi-Controller + SPMD 范式适合简单的代码逻辑（所有 GPU 执行一样的代码）。随着深度学习分布式系统的代码越来越复杂（Tensor Parallel、Pipeline Parallel、MoE Parallel 和强化学习训练流程），Single-Controller 易用性的优势日渐凸显。
