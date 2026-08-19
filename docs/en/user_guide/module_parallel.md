# Module Parallel: Implementing DP/TP/CP/PP Combinations

DTorch's `Module` system is completely consistent with PyTorch in terms of interfaces and usage — model code written by users runs in DTorch without any modification. On top of that, to support distribution, DTorch adds a small set of extension capabilities to `Module` (subclasses of `nn.Module`), natively supporting combinations of **Data Parallel, Tensor Parallel, Context Parallel, Pipeline Parallel** while keeping the single-GPU coding style. This article uses `Linear` as an example to explain the parallel mechanisms at the Module level; see [Llama Parallel Example](llama_parallel.md) for the complete DP + TP + PP + CP example of the Llama model.

Prerequisites: the DTensor and `redistribute()` sections of [Python API Overview](python_api_overview.md).

---

## 1. Module's redistribute hooks

The `Module` base class provides the `redistribute_input()` and `redistribute_output()` hooks, which are automatically called before and after `forward` executes. Subclasses override these two methods to implement **transparent input/output redistribution** — this is the unified mechanism by which the later `Linear` subclasses and complete models (Llama) build their parallel logic.

**Base class interface (`python/dtorch/nn/modules/module.py`):**

```python
class Module:
    def redistribute_input(self, *args, **kwargs):
        """May be overridden by subclasses; returns the (args, kwargs) tuple"""
        return args, kwargs

    def redistribute_output(self, output):
        """May be overridden by subclasses; returns the redistributed output"""
        return output

    def __call__(self, *args, **kwargs):
        # 1. call redistribute_input to redistribute the inputs
        args, kwargs = self.redistribute_input(*args, **kwargs)
        # 2. execute forward
        output = self.forward(*args, **kwargs)
        # 3. call redistribute_output to redistribute the output
        output = self.redistribute_output(output)
        return output
```

Typical usage: in `redistribute_input`, convert the inputs to the distribution the model expects and save the original distribution; in `redistribute_output`, restore the output to the original distribution, thus staying transparent to the caller.

---

## 2. Implementing DP / TP / CP / PP Parallel

DTorch expresses all kinds of parallel strategies uniformly through the named dimensions of `DeviceMesh` — give each dimension a semantic name (`"dp"`, `"tp"`, `"cp"`, `"pp"`) and declare the distribution on each dimension in the `Placements` of Tensors and Parameters; the framework automatically inserts collective communication accordingly. The four kinds of parallelism at the Module level are described below.

### Data Parallel

Data parallel shards the input by batch on the `"dp"` dimension, and the weights stay `Replicate()` on the `"dp"` dimension. Just declare a dimension named `"dp"` in the `DeviceMesh` and shard the input by batch to that dimension at the model entry:

```python
device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=["dp"])

# shard the input by batch to the dp dimension
input = input.redistribute_by_dict(device_mesh, placements_dict={"dp": Shard(0)})
```

Weights are `Replicate()` by default on all non-`"tp"` dimensions (see Section 3, Linear implementation walkthrough), so DP needs no extra sharding — each device keeps identical weights.

### Tensor Parallel

Tensor parallel shards weights on the `"tp"` dimension. **Only two kinds of layers actually need sharding — `Linear` and `Embedding`**: the former via the `ColumnParallelLinear` / `RowParallelLinear` subclasses, the latter via `EmbeddingWithReplicateOutput`. These subclasses have the weight sharding and input/output validation/conversion built in.

```python
device_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=["tp"])

fc1 = nn.ColumnParallelLinear(hidden_size, intermediate_size)                 # weight Shard(0) on the tp dim
fc2 = nn.RowParallelLinearWithReplicateOutput(intermediate_size, hidden_size) # output auto AllReduce'd to Replicate

embed = nn.EmbeddingWithReplicateOutput(vocab_size, hidden_size)              # sharded by embedding_dim, output AllGather'd

# called exactly like ordinary nn.Linear / nn.Embedding (token_ids must be Replicate on the tp dim)
h = embed(token_ids)  # -> Replicate (already AllGather'd)
h = fc1(h)            # ColumnParallel: Replicate in, Shard along the feature dim out
h = fc2(h)            # RowParallel: Shard in, AllReduce then Replicate out
```

- `Linear`: `ColumnParallelLinear` shards by the output dimension `Shard(0)`; `RowParallelLinearWithReplicateOutput` shards by the input dimension `Shard(1)` and AllReduces at the output to `Replicate`.
- `Embedding`: weights are sharded by `embedding_dim` (`Shard(1)`); `EmbeddingWithReplicateOutput` AllGathers at the output so the result is `Replicate` again on the tp dimension.

> Unlike Megatron-LM, DTorch does not require manually sharding weights by rank and loading the corresponding shards separately: after declaring `Placements`, DTensor automatically loads the Tensor and shards it by dimension internally.

### Context Parallel

Context Parallel shards Q/K/V by **sequence length** on the `"cp"` dimension, specifically for long-sequence attention. DTorch supports two CP variants, distinguished by the `DeviceMesh` dimension name:

- **Ulysses CP** (dimension name `"ulysess_cp"`): shards by attention head, reorganized internally with all-to-all.
- **Ring CP** (dimension name `"ring_cp"`): performs ring attention communication on the sequence dimension.

Both variants require Q/K/V to be `Shard(2)` on the CP dimension (i.e., the sequence dimension `L` in the `[N, H, L, E]` layout). As long as a `ulysess_cp` / `ring_cp` dimension exists in the `DeviceMesh`, `scaled_dot_product_attention` automatically enables the corresponding CP implementation:

```python
import dtorch
import dtorch.nn.functional as F
from dtorch import init_device_mesh, Shard

device_mesh = init_device_mesh(
    "cuda", (dp, ulysess_cp, ring_cp),
    mesh_dim_names=["dp", "ulysess_cp", "ring_cp"],
)

# Q/K/V Shard(2) on the cp dims: sharded by the sequence dim L; dp dim sharded by batch Shard(0)
placements = [Shard(0), Shard(2), Shard(2)]
query = dtorch.randn(N, H, L, E, device_mesh=device_mesh, placements=placements)
key   = dtorch.randn(N, H, S, E, device_mesh=device_mesh, placements=placements)
value = dtorch.randn(N, H, S, E, device_mesh=device_mesh, placements=placements)

# CP is enabled automatically when the DeviceMesh has ulysess_cp / ring_cp dimensions
out = F.scaled_dot_product_attention(query, key, value, is_causal=True)

print(out.device_mesh)   # DeviceMesh('cuda', dim_name: ['dp', 'ulysess_cp', 'ring_cp'], shape: (2, 2, 2), data: ...)
print(out.placements)    # [Shard(0), Shard(2), Shard(2)]
```

> Implementation details: see `python/dtorch/nn/scaled_dot_product_attention_with_cp.py`.

### Pipeline Parallel

Pipeline Parallel (PP) divides the different layers of a model into multiple stages (devices), and activations are passed between adjacent stages via `redistribute`. DTorch natively supports PP at the Module level — just bind each sub-Module to the `DeviceMesh` of its stage, and a single model definition is kept; no manual model splitting/trimming.

The three core tools:

- `device_mesh.unbind("pp")`: unfolds the `"pp"` dimension into several sub-`DeviceMesh`es, one per stage (i.e., sub-meshes with the `"pp"` dimension removed and the other dimensions unchanged; e.g., a `dp×tp×pp` mesh yields a set of `dp×tp` stage meshes). When the DeviceMesh has no `"pp"` dimension, it returns `[device_mesh]`, degenerating to a single stage.
- `assign_layers_to_stages(num_layers, num_stages)`: evenly maps `num_layers` layers to `num_stages` stages, returning a list of length `num_layers` whose *i*-th entry is the stage number of the *i*-th layer (when the division is uneven, earlier stages get one extra layer).
- `Graph.default_graph().device_mesh_guard(stage_mesh)`: a context manager that binds sub-Modules created inside it to the specified stage's devices.

```python
import dtorch
from dtorch import nn, Graph, DeviceMesh, init_device_mesh, assign_layers_to_stages

class Transformer(nn.Module):
    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        num_layers = 4

        # 1. unfold the pp dimension to get each stage's DeviceMesh (sub-mesh without the "pp" dim)
        self.pp_stage_meshes = device_mesh.unbind("pp")
        pp_stages = len(self.pp_stage_meshes)
        # 2. evenly map each layer to a stage, getting each layer's stage number
        self.layer_stage_ids = assign_layers_to_stages(num_layers, pp_stages)

        # 3. bind each sub-Module to the DeviceMesh of its stage
        with Graph.default_graph().device_mesh_guard(self.pp_stage_meshes[0]):
            self.tok_embeddings = nn.EmbeddingWithReplicateOutput(vocab_size, hidden_size)

        self.layers = nn.ModuleList()
        for layer_id in range(num_layers):
            layer_device_mesh = self.pp_stage_meshes[self.layer_stage_ids[layer_id]]
            with Graph.default_graph().device_mesh_guard(layer_device_mesh):
                self.layers.append(TransformerBlock(...))

        with Graph.default_graph().device_mesh_guard(self.pp_stage_meshes[-1]):
            self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: dtorch.Tensor):
        h = self.tok_embeddings(tokens)

        for layer_id, layer in enumerate(self.layers):
            layer_device_mesh = self.pp_stage_meshes[self.layer_stage_ids[layer_id]]
            h = h.redistribute(device_mesh=layer_device_mesh)   # activations moved automatically across stages
            h = layer(h, self.freqs_cis)

        output = self.output(h).float()
        return output

device_mesh = init_device_mesh("cuda", (dp, tp, pp), mesh_dim_names=["dp", "tp", "pp"])
model = Transformer(device_mesh)
x = dtorch.randn(batch_size, in_dim, device="cuda")
y = model(x)
```

In `forward`, `h.redistribute(device_mesh=layer_device_mesh)` moves activations between adjacent stages; when the target layer is on the same stage as the current one, this operation performs no actual communication.

---

## 3. Linear implementation walkthrough

DTorch's `Linear` module ([`source`](https://github.com/tingkuanpei/dtorch/blob/main/python/dtorch/nn/modules/linear.py)) natively supports DP, TP, CP and other parallel strategies. Its core principle is: **only the TP dimension needs to shard the Weight; on the DP and CP dimensions the Weight always stays fully replicated (`Replicate()`)**.

### Core parameters: tp_dim and tp_shard_type

The `Linear` constructor signature:

```python
Linear(in_features, out_features, bias=True, device=None, dtype=None,
       device_mesh=None, *, tp_dim="tp", tp_shard_type=None)
```

**`tp_dim`** — specifies which dimension of the DeviceMesh to perform tensor parallel (TP) weight sharding on:

| Value type | Meaning | Example |
|---|---|---|
| `str` (default `"tp"`) | matches the dimension with the same name in `device_mesh.dim_names` | `tp_dim="tp"` → shard on the dimension named `"tp"` |
| `int` | directly specifies the DeviceMesh dimension index | `tp_dim=1` → shard on dimension 1 |
| `None` | no TP sharding, all weights stay Replicate | `ReplicateParallelLinear` sets `tp_dim=None` |

> **Key behavior**: when `tp_dim` is a string, `device_mesh.dim_name_index(tp_dim)` is called to look up the matching dimension. **If the DeviceMesh has no dimension with that name** (e.g., the DeviceMesh only has `"dp"` and `"cp"` but no `"tp"`), it returns `None`, and **no TP sharding is performed**.

**`tp_shard_type`** — specifies the sharding direction of the weight:

| tp_shard_type | weight Placement (on tp_dim) | bias Placement (on tp_dim) | Meaning |
|---|---|---|---|
| `"col"` | `Shard(0)` | `Shard(0)` | shard by output features, each device holds part of the output columns |
| `"row"` | `Shard(1)` | `Partial()` | shard by input features, each device holds part of the input rows |

### Weight sharding rules

At `Linear` initialization, the initial Placements of all weights and bias are `Replicate()` on **all dimensions**. Only the dimension corresponding to `tp_dim` is replaced with a sharding Placement:

```python
weight_placements = [Replicate()] * device_mesh.ndim   # all dimensions initially Replicate
bias_placements   = [Replicate()] * device_mesh.ndim

if tp_dim is not None:
    weight_placements[tp_dim] = Shard(1) if tp_shard_type == "row" else Shard(0)
    bias_placements[tp_dim]   = Partial()  if tp_shard_type == "row" else Shard(0)
```

**This naturally guarantees compatibility with the DP, CP and other dimensions**: because only the dimension matched by `tp_dim` is sharded, the remaining dimensions (such as `"dp"`, `"cp"`) always stay `Replicate()` and are unaffected by the TP logic.

```
Assume DeviceMesh dim_names = ["dp", "tp", "cp"], tp_dim="tp"

          dp dim         tp dim         cp dim
weight: [Replicate(), Shard(0|1),  Replicate()]
          ↑ data parallel  ↑ TP shard   ↑ context parallel
          full copy        only modified  full copy
```

### redistribute_input / redistribute_output — input/output validation and conversion

`Linear` calls `redistribute_input` and `redistribute_output` before and after `forward` respectively, validating the input and output Tensors and performing optional Placements conversion. Both operate only on the dimension corresponding to `tp_dim`; the other dimensions stay unchanged via `default_placement_mode="keep"`.

**redistribute_input** — performs two tasks:

1. **Optional conversion**: if the caller passes `input_placement`, redistribute the input on `tp_dim` to the target Placement.
2. **Validation**: assert that the input's Placement on `tp_dim` matches the weight sharding method.

```python
def redistribute_input(self, input, input_placement=None):
    if self.tp_dim is not None and input_placement is not None:
        input = input.redistribute_by_dict(
            placements_dict={self.tp_dim: input_placement},
            default_placement_mode="keep",
        )
    if self.tp_dim is not None:
        expect = Shard(input.dim() - 1) if self.tp_shard_type == "row" else Replicate()
        assert input.check_placement(self.tp_dim, expect)
    return [input], {}
```

| tp_shard_type | required input Placement (on tp_dim) | Reason |
|---|---|---|
| `"col"` | `Replicate()` | each device needs the complete input to compute its partial output |
| `"row"` | `Shard(input.ndim - 1)` | the input is sharded on the hidden dimension, aligned with the weight's in_features sharding |

**redistribute_output** — converts the output to the specified Placement on `tp_dim`. The base class performs no conversion by default (returns directly when `output_placement=None`); subclasses pass a specific value to enable automatic conversion:

```python
def redistribute_output(self, output, output_placement=None):
    if self.tp_dim is not None and output_placement is not None:
        output = output.redistribute_by_dict(
            placements_dict={self.tp_dim: output_placement},
            default_placement_mode="keep",
        )
    return output
```

For example, `RowParallelLinearWithReplicateOutput` calls `redistribute_output(output, Replicate())`, automatically inserting an AllReduce to convert to `Replicate()` after RowParallel produces a `Partial()` output.

### Convenience subclasses

Based on the combinations of `tp_dim` and `tp_shard_type`, DTorch provides the following preset subclasses covering common parallel scenarios:

| Class | tp_dim | tp_shard_type | redistribute_input | redistribute_output |
|---|---|---|---|---|
| `ColumnParallelLinear` | `"tp"` | `"col"` | validates input is Replicate | no conversion |
| `ColumnParallelLinearWithReplicateOutput` | `"tp"` | `"col"` | validates input is Replicate | output converted to Replicate |
| `ColumnParallelLinearWithReplicateInputOutput` | `"tp"` | `"col"` | input converted to Replicate | output converted to Replicate |
| `RowParallelLinear` | `"tp"` | `"row"` | validates input is Shard(-1) | no conversion |
| `RowParallelLinearWithReplicateOutput` | `"tp"` | `"row"` | validates input is Shard(-1) | output converted to Replicate |
| `ReplicateParallelLinear` | `None` | — | no validation | no conversion |

The most commonly used combination is `ColumnParallelLinear` + `RowParallelLinearWithReplicateOutput` (see [Llama Parallel Example](llama_parallel.md)).
