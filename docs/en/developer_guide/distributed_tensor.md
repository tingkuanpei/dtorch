# Distributed Tensor (DTensor)

The basic concepts of DTensor, the common APIs and the Module-level parallel usage have been introduced in the user guide. This article supplements three advanced topics from the **developer's perspective**: **how Operators infer the output distribution through PlacementSignature** (and the design principle of "no automatic redistribute injection"), the internal handling of **uneven sharding**, and the **automatic sharding when loading state_dict and the automatic gathering when retrieving values**.

Prerequisites:

- Basic concepts (DeviceMesh, Placements, the three Placement strategies, how to read multi-dimensional distribution): see [Distributed Tensor Overview](../user_guide/distributed_tensor_overview.md)
- Common APIs for DTensor creation, operator inference, `redistribute`, etc.: see [Python API Overview](../user_guide/python_api_overview.md)
- Module-level usage of DP/TP/CP/PP: see [Module Parallel](../user_guide/module_parallel.md)
- A complete DP+TP+PP+CP example with the Llama model: see [Llama Parallel Example](../user_guide/llama_parallel.md)

> PyTorch also supports DTensor, but it is still in the [alpha stage](https://docs.pytorch.org/docs/stable/distributed.tensor.html).

## 1. How Operators handle DTensor

For the basic behavior — operators natively supporting DTensor, automatic inference of output DeviceMesh/Placements, and exceptions raised when Placements are incompatible — see the user guide [Python API Overview](../user_guide/python_api_overview.md). This section dives into the underlying mechanism and design principles.

**PlacementSignature — automatic inference of output Placements**

Every Operator has a built-in **PlacementSignature** rule table declaring the mapping between input and output Placements. The framework automatically matches rules against the input DTensors' actual Placements and infers the output Placements (this process **only infers metadata and triggers no actual communication**).

**Core principle: no automatic redistribute injection**

> When the input Placements cannot match the signature rules, DTorch **raises an error by default**, asking the user to adjust the code.

Design rationale: `tensor.redistribute()` is an expensive operation (it involves collective communications such as all-gather, all-to-all, all-reduce at the bottom). If the framework performed it implicitly, users would lose awareness of, and the room to optimize, the communication cost. Therefore users need to **actively care about the Placements and DeviceMesh of every Tensor, and explicitly call `tensor.redistribute()` when necessary**.

**Exception: a few operators inject communication automatically**

To balance code simplicity, a few operators **automatically insert redistribute** to simplify common scenarios. For example, in `BroadcastBinaryOp` (binary operations such as add/subtract/multiply/divide), when a Replicate Tensor is computed with a Shard Tensor, the framework automatically converts the Replicate side to Shard (see `PlacementR2S` in `dtorch/api/cpp/functional/implement/broadcast_op_imlp.cc`). Operators with such automatically injected communication are highlighted in their respective documents.

> More details: see [PlacementSignature](https://tingkuanpei.github.io/dtorch/cn/developer_guide/operator/placement_signature/).

## 2. Uneven sharding

DTorch natively supports uneven sharding of Tensors — when a dimension's length is not divisible by the device count, the framework **automatically computes each device's local Shape** (rather than raising an error or requiring the user to adjust manually).

```python
device_mesh = dtorch.DeviceMesh("cpu", [0, 1, 2, 3])

# shape [4, 11] sharded along dim 1 onto 4 devices: 11 % 4 ≠ 0, uneven sharding
x = dtorch.randn(4, 11, device_mesh=device_mesh, placements=[Shard(1)])
# GPU 0: local shape [4, 3]
# GPU 1: local shape [4, 3]
# GPU 2: local shape [4, 3]
# GPU 3: local shape [4, 2]  (1 column less)
```

**Automatic padding during redistribute**

Collective communication libraries such as NCCL require the input Tensor shapes of all ranks to be identical. Therefore, when executing `tensor.redistribute()`, DTorch **automatically inserts padding to align** unevenly sharded Tensors, and removes the padding after the communication completes — the whole process is transparent to the user. For example, when redistributing the `[Shard(1)]` Tensor above to `[Replicate()]`, the framework first fills the gap on GPU 3 and then executes the all-gather.

```python
# redistribute handles padding / unpadding automatically
x_r = x.redistribute(device_mesh=device_mesh, placements=[Replicate()])
# the user does not need to be aware of the internal padding logic
```

## 3. Automatic sharding and automatic gathering

The distribution of DTensors is transparent to the user: **weights are sharded automatically on load, and values are gathered automatically on retrieval** — the two are inverse operations of each other.

**Loading state_dict — automatic sharding**

The weights in `state_dict` are complete `torch.Tensor`s, while the target Parameters already carry Placements. The framework automatically shards the complete weights to each rank according to their Placements: `Shard(dim)` shards along the corresponding dimension, `Replicate` copies to every rank, and `Partial` keeps the real value only on rank 0. Users load one complete state_dict, without needing to pre-shard weights by rank as in Megatron-LM.

**Retrieving values — automatic gathering**

When calling `to_torch()` / `to_torch_async()` to retrieve values, the framework automatically gathers the shards on all ranks back into one complete Tensor: `Shard(dim)` concatenates along the corresponding dimension, `Replicate` takes any one copy, and `Partial` performs element-wise summation. Users do not need to call all-gather manually.
