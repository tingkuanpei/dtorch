# Distributed Tensor Overview

A DTensor (Distributed Tensor) describes "how a Tensor is sharded and distributed across multiple devices". Compared to a regular Tensor, a DTensor has two extra attributes: **DeviceMesh** and **Placements**.

- **DeviceMesh**: an n-dimensional array describing the topology of devices in the cluster.
- **Placements**: a 1D array of length n, describing the sharding strategy of the Tensor on each dimension of the DeviceMesh — `Shard(dim)` (shard) / `Replicate()` (replicate) / `Partial()` (partial reduction).

The placement sharding strategies are as follows:

| Placement | Meaning |  Result |
|-----------|------|------|
| `Replicate()` | Fully replicated on all devices | Each device holds the full data |
| `Shard(dim)` | Evenly sharded along dimension `dim` across devices | Each device holds 1/N of the data |
| `Partial()` | Each device holds a partial result; reduce sum is needed to get the full value | Each device holds a partial sum |

## Simple Example

Take a Tensor of shape=(4, 4) distributed across 2 GPUs:

<figure markdown>
  ![DTensor concept diagram](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/dtensor_en.png)
  <figcaption>Figure 1: DeviceMesh and Placements of a DTensor</figcaption>
</figure>

## Multi-Dimensional Distribution

DeviceMesh supports multiple dimensions — **each dimension corresponds to one parallel strategy**, enabling combinations of multiple parallelisms. Placements is a sequence whose length equals the mesh dimension count n, where the i-th Placement describes how the Tensor is distributed along the i-th mesh dimension.

### Example: DP + TP

8 GPUs, running Data Parallel (2-way) and Tensor Parallel (4-way) simultaneously. Use a 2D mesh:

```python
from dtorch import init_device_mesh, Replicate, Shard

# 2×4 = 8 GPUs: first dim "dp", second dim "tp"
device_mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=["dp", "tp"])
# The two Placements correspond to the dp and tp dimensions: dp: replicate, tp: shard along dim 1
placements=[Replicate(), Shard(1)]

weight = dtorch.randn(8, 128, device_mesh=device_mesh, placements=placements)
```

How to read this:

- **dp dimension (2 GPUs) → `Replicate()`**: each of the two groups holds a full copy of the Tensor and computes independently (data parallel).
- **tp dimension (4 GPUs) → `Shard(1)`**: the 4 GPUs in each group shard the Tensor into 4 pieces along the hidden dimension (128); only combined is it complete (tensor parallel).
