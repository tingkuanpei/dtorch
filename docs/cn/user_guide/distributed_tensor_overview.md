# Distributed Tensor Overview

DTensor(Distributed Tensor) 是描述“一个 Tensor 如何被切分并分布到多个设备上”的描述方法。DTensor 比普通 Tensor 多了两个属性： **DeviceMesh** 和 **Placements**。

- **DeviceMesh**：n-dimensional array，描述集群中 device 的拓扑。
- **Placements**：1D array of length n。描述 Tensor 在 DeviceMesh 每一维上的切分策略——`Shard(dim)`（切分）/ `Replicate()`（复制）/ `Partial()`（部分聚合）。

Placement 的切分策略如下：

| Placement | 含义 |  结果 |
|-----------|------|------|
| `Replicate()` | 完整复制到所有设备 | 每个设备持有完整数据 |
| `Shard(dim)` | 沿第 `dim` 轴均匀切分到各设备 | 每设备持有 1/N 数据 |
| `Partial()` | 各设备持有部分结果，需 reduce sum 得到完整值 | 每设备持有部分和 |

## 简单示例

以 shape=(4, 4) 分布在 2卡上的 Tensor 为例：

<figure markdown>
  ![DTensor 概念图](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/dtensor_cn.png)
  <figcaption>图 1：DTensor 的 DeviceMesh 与 Placements 示意</figcaption>
</figure>

## 多维分布

DeviceMesh 支持多维，**每一维对应一种并行策略**，从而实现多种并行的组合。Placements 为长度等于 mesh 维数 n 的序列，其中第 `i` 个 Placement 描述 Tensor 沿第 `i` 个 mesh 维度的分布方式。

### 示例：DP + TP

8 张卡，同时做 Data Parallel（2 路）和 Tensor Parallel（4 路）。用一个 2D mesh：

```python
from dtorch import init_device_mesh, Replicate, Shard

# 2×4 = 8 卡：第一维 "dp"，第二维 "tp"
device_mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=["dp", "tp"])
# 两个 Placement 分别对应 dp、tp 两个维度, dp: 复制  tp: 沿 dim 1 切分
placements=[Replicate(), Shard(1)]

weight = dtorch.randn(8, 128, device_mesh=device_mesh, placements=placements)
```

读法：

- **dp 维度（2 卡）→ `Replicate()`**：两组各持一份完整的 Tensor，各算各的（数据并行）。
- **tp 维度（4 卡）→ `Shard(1)`**：每组内的 4 张卡沿 hidden 维度（128）把 Tensor 切成 4 份，拼起来才完整（张量并行）。
