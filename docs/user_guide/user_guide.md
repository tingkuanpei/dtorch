# User Guide

本指南分两步带你上手 DTorch：先理解分布式张量（DTensor）的核心概念，再学习如何用 DTorch Python API 编写分布式程序。

## 推荐阅读顺序

1. **[DTensor 概览](distributed_tensor_overview.md)** — 了解 `DeviceMesh` 与 `Placements` 如何描述张量在多设备上的分布。这是 DTorch 全部分布式能力的基石，建议先读。
2. **[Python API 概览](python_api_overview.md)** — 在单卡 PyTorch 代码的基础上，只需声明 `device_mesh` 和 `placements`，即可扩展为多卡分布式程序。
3. **[Module 并行](module_parallel.md)** — `Linear` 与 `nn.Module` 如何组合实现 DP / TP / CP，附 Llama 完整示例。
