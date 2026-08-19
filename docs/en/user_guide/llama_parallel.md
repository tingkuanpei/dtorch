# Llama Parallel Example: DP + TP + PP + CP

This article uses the Llama model as an example to show how to implement any combination of **Data Parallel, Tensor Parallel, Pipeline Parallel, Context Parallel** with DTorch's `DeviceMesh` and `Module` system. Full code: [`python/dtorch/test/modules/llama.py`](https://github.com/tingkuanpei/dtorch/blob/main/python/dtorch/test/modules/llama.py); tests: [`python/dtorch/test/modules/test_llama.py`](https://github.com/tingkuanpei/dtorch/blob/main/python/dtorch/test/modules/test_llama.py).

Prerequisites: [Python API Overview](python_api_overview.md) (DTensor and `redistribute()`) and [Module Parallel](module_parallel.md) (the redistribute hooks and the Linear parallel subclasses).

---

## 1. DeviceMesh: five named dimensions

Llama uses a `DeviceMesh` with up to five dimensions, one named dimension per kind of parallelism:

```python
device_mesh = init_device_mesh(
    "cuda",
    (dp, tp, pp, ulysess_cp, ring_cp),
    mesh_dim_names=["dp", "tp", "pp", "ulysess_cp", "ring_cp"],
)
```

| Dimension name | Parallel type | Role |
|---|---|---|
| `dp` | Data Parallel | shard the input by batch (dim 0), weights fully replicated |
| `tp` | Tensor Parallel | shard `Linear` / `Embedding` weights (see [Module Parallel](module_parallel.md)) |
| `pp` | Pipeline Parallel | divide the different layers into multiple stages |
| `ulysess_cp` | Context Parallel (Ulysses) | shard Q/K/V by the sequence dim, all-to-all reorganization of heads |
| `ring_cp` | Context Parallel (Ring) | shard Q/K/V by the sequence dim, ring attention communication |

Setting any dimension to 1 degenerates to no parallelism on that dimension; the model code is completely transparent to the dimension combination — no rewriting for different strategies.

---

## 2. Top-level model: LlamaForCausalLM

The top-level model validates the DeviceMesh dimension names, redistributes the input to the distribution the model expects, and restores the original distribution at the output:

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config, device_mesh=None):
        super().__init__()
        device_mesh = get_default_device_mesh(device_mesh=device_mesh)
        # only the declared dimension names are allowed
        device_mesh.check_all_dim_names_in_set({"dp", "tp", "pp", "ulysess_cp", "ring_cp"})

        self.model = LlamaModel(config, device_mesh)
        # lm_head lives on the last pipeline stage
        with Graph.default_graph().device_mesh_guard(self.model.pp_stage_meshes[-1]):
            self.lm_head = nn.ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False,
            )

    def redistribute_input(self, input_ids):
        # save the input's original mesh/placements, restored at the output
        self.input_device_mesh = input_ids.device_mesh
        self.input_placement = input_ids.placements

        # redistribute to the first stage (embedding) mesh:
        #   dp         -> Shard(0)    shard by batch
        #   tp         -> Replicate() TP is handled by weight sharding, input needs no sharding
        #   ulysess_cp / ring_cp -> Shard(1)  shard by the sequence dim
        # Dimension names absent from the target mesh are ignored automatically, so they can be listed unconditionally.
        input_ids = input_ids.redistribute_by_dict(
            self.first_param_device_mesh(),
            placements_dict={
                "dp": Shard(0),
                "tp": Replicate(),
                "ulysess_cp": Shard(1),
                "ring_cp": Shard(1),
            },
        )
        return [input_ids], {}

    def redistribute_output(self, logits):
        # restore the input's original distribution, staying transparent to the caller
        return logits.redistribute(self.input_device_mesh, placements=self.input_placement)
```

Key points:

- `check_all_dim_names_in_set(...)` ensures the DeviceMesh dimension names stay within the supported range
- The dimension names in `placements_dict` do **not need to be checked for existence in advance**: an entry is ignored when the target mesh has no such dimension
- `redistribute_input` moves the input to the first stage's mesh; `redistribute_output` restores the caller's original distribution from the last stage (where lm_head lives)

---

## 3. Data Parallel: sharding input by batch

Unlike TP/PP/CP, DP involves no model structure at all — it is entirely expressed by the **distribution of the data**: the input is sharded by batch as `Shard(0)` on the `dp` dimension, and the weights stay `Replicate()` (fully replicated) on the `dp` dimension.

- On the input side, `"dp": Shard(0)` in `redistribute_input` is all the DP-related code in the model: after sharding, each dp replica independently executes the full forward on its own slice of the batch, with no communication on the `dp` dimension at all.
- On the output side, `redistribute_output` restores the caller's original distribution:

```python
# input side
placements_dict={
    "dp": Shard(0),        # each dp replica processes one slice of the batch
    "tp": Replicate(),
    ...
}

# output restored to the caller's distribution: Replicate caller -> AllGather along the dp dim
logits = logits.redistribute(self.input_device_mesh, placements=self.input_placement)
```

---

## 4. Pipeline Parallel: mapping layers to stages

`LlamaModel` receives the complete `device_mesh`, internally unfolds each stage's sub-mesh with `unbind("pp")`, and binds each layer to its stage. `assign_layers_to_stages(num_layers, num_stages)` distributes layers to stages as evenly as possible (when the division is uneven, earlier stages get one extra layer); for example, `assign_layers_to_stages(4, 2)` returns `[0, 0, 1, 1]`.

```python
class LlamaModel(nn.Module):
    def __init__(self, config, device_mesh):
        super().__init__()
        # unbind("pp") unfolds the pp dim into a set of sub-meshes (pp dim removed, other dims unchanged);
        # with no pp dim it returns [device_mesh], degenerating to a single stage.
        self.pp_stage_meshes = device_mesh.unbind("pp")
        # evenly map the num_hidden_layers layers to the stages, getting each layer's stage number
        self.layer_stage_ids = assign_layers_to_stages(
            config.num_hidden_layers, len(self.pp_stage_meshes)
        )

        # embedding and rotary live on the first stage
        with Graph.default_graph().device_mesh_guard(self.pp_stage_meshes[0]):
            self.rotary_emb = LlamaRotaryEmbedding(config=config)
            self.embed_tokens = nn.EmbeddingWithReplicateOutput(config.vocab_size, config.hidden_size)

        # each decoder layer is bound to the mesh of its stage
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            this_stage_device_mesh = self.pp_stage_meshes[self.layer_stage_ids[layer_idx]]
            with Graph.default_graph().device_mesh_guard(this_stage_device_mesh):
                self.layers.append(LlamaDecoderLayer(config, layer_idx))

        # the final norm lives on the last stage
        with Graph.default_graph().device_mesh_guard(self.pp_stage_meshes[-1]):
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

In `forward`, layers execute in order; before entering each layer the activations are redistributed to that layer's stage mesh, and activations are moved automatically across stages:

```python
def forward(self, input_ids=None):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    position_ids = dtorch.arange(
        inputs_embeds.shape[1], device_mesh=hidden_states.device_mesh
    ).unsqueeze(0)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(self.layers):
        this_stage_device_mesh = self.pp_stage_meshes[self.layer_stage_ids[layer_idx]]
        # both hidden_states and position_embeddings are redistributed to this_stage_device_mesh
        hidden_states = hidden_states.redistribute(
            device_mesh=this_stage_device_mesh, placements=hidden_states.placements
        )
        position_embeddings = [
            pe.redistribute(device_mesh=this_stage_device_mesh, placements=pe.placements)
            for pe in position_embeddings
        ]
        hidden_states = decoder_layer(
            hidden_states, position_embeddings=position_embeddings,
        )

    hidden_states = hidden_states.redistribute(
        device_mesh=self.pp_stage_meshes[-1], placements=hidden_states.placements
    )
    hidden_states = self.norm(hidden_states)
    return hidden_states
```

---

## 5. Context Parallel: RoPE and sequence sharding

Like DP, CP shards no weights — the sharding is entirely expressed by the data distribution, starting from the `placements_dict` in `redistribute_input`: `input_ids` is sharded by the sequence dimension as `Shard(1)` on the `ulysess_cp` / `ring_cp` dimensions (the `L` dim in the `[N, L]` layout):

```python
placements_dict={
    "dp": Shard(0),
    "tp": Replicate(),
    "ulysess_cp": Shard(1),   # each CP rank holds one slice of the sequence
    "ring_cp": Shard(1),
}
```

The sequence sharding propagates naturally through the model with `Shard(1)`: the embedding output `hidden_states` is `[N, L, E]` (sharded on `L`); after the Q/K/V projections, `view(bsz, q_len, -1, head_dim).transpose(1, 2)` converts the layout to `[N, H, L, E]`, and the sequence-dimension `Shard` moves to dim 2 accordingly — the Q/K/V `Shard(2)` that attention requires (the `L` dim in the `[N, H, L, E]` layout) is obtained automatically from the input sharding and the layout transformation, with no explicit `redistribute` needed.

As long as `ulysess_cp` / `ring_cp` dimensions exist in the mesh, `scaled_dot_product_attention` automatically enables the corresponding CP implementation — the attention layer needs no extra code.

The position encodings likewise need **no extra handling**: when multiplied with the CP-sharded Q/K, the broadcast binary operator (`broadcast_op_imlp.cc`) automatically converts the `Replicate` side to `Shard`, aligning with the Q/K distribution.

---

## 6. Tensor Parallel: Attention and MLP

TP reuses the `ColumnParallelLinear` + `RowParallelLinearWithReplicateOutput` combination introduced in [Module Parallel](module_parallel.md); adding CP/PP does not affect any weight sharding logic:

```python
class LlamaSdpaAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        ...
        # ColumnParallel: sharded by the output columns (attention heads) on the tp dim
        self.q_proj = nn.ColumnParallelLinear(self.hidden_size, self.num_heads * self.head_dim, bias=...)
        self.k_proj = nn.ColumnParallelLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=...)
        self.v_proj = nn.ColumnParallelLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=...)
        # RowParallel: sharded by the input dim, output auto AllReduce'd to Replicate
        self.o_proj = nn.RowParallelLinearWithReplicateOutput(self.num_heads * self.head_dim, self.hidden_size, bias=...)

    def forward(self, hidden_states, position_embeddings):
        ...
        # CP is enabled automatically when the mesh has ulysess_cp / ring_cp dimensions
        attn_output = dtorch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True,
        )
        ...
```

MLP and Embedding follow the same pattern:

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.ColumnParallelLinear(self.hidden_size, self.intermediate_size, bias=...)
        self.up_proj   = nn.ColumnParallelLinear(...)
        self.down_proj = nn.RowParallelLinearWithReplicateOutput(self.intermediate_size, self.hidden_size, bias=...)

# Embedding sharded by embedding_dim, output AllGather'd to Replicate
self.embed_tokens = nn.EmbeddingWithReplicateOutput(config.vocab_size, config.hidden_size)
```

---

## 7. Summary: roles of the four parallel dimensions

| Component | dp | tp | pp | cp |
|---|---|---|---|---|
| input ids | `Shard(0)` (batch) | `Replicate()` | on the first stage | `Shard(1)` (sequence) |
| `Linear` / `Embedding` weights | `Replicate()` | `Shard(0|1)` | each stage holds its own layers | `Replicate()` |
| Q/K/V | follows the input | follows ColumnParallel output | — | `Shard(2)` (sequence) |
| cos/sin | `Replicate()` | `Replicate()` | moved across stages with activations | `Replicate()` (broadcast op auto-converts to `Shard`) |

---

## 8. Testing: switch parallel strategies with one call

The test uses a single-GPU PyTorch model as reference and combines parallel strategies arbitrarily through a `dtorch_imp` closure (see `python/dtorch/test/modules/test_llama.py`):

```python
def _test_llama(test_case, device):
    torch_out = torch_llama(torch_in)[0]   # PyTorch reference output

    def dtorch_imp(dp=1, tp=1, pp=1, ulysess_cp=1, ring_cp=1):
        device_mesh = init_device_mesh(
            device,
            (dp, tp, pp, ulysess_cp, ring_cp),
            mesh_dim_names=["dp", "tp", "pp", "ulysess_cp", "ring_cp"],
        )
        if not is_graph_satisfy(dtorch.default_graph, device_mesh):
            return
        dtorch_in = dtorch.Tensor(torch_in)
        dtorch_llama = LlamaForCausalLM(config, device_mesh=device_mesh)
        dtorch_llama.load_state_dict(torch_llama.state_dict())
        dtorch_out = dtorch_llama(dtorch_in)
        assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)

    dtorch_imp(dp=2)
    dtorch_imp(dp=2, tp=2, pp=2, ulysess_cp=2)   # DP+TP+PP+CP combination
```

The model code is unaware of the concrete strategy: any combination just changes the shape and dimension names of the `DeviceMesh`; `is_graph_satisfy` automatically skips when the cluster does not satisfy the mesh requirement, so the test runs in environments of any size.

The test command: without a multi-GPU cluster, you can also enable [single-device distributed simulation](python_api_overview.md#6-easter-egg-single-device-distributed-simulation) to run all strategy combinations on a single GPU:

```bash
# `DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE=16` simulates with 16 virtual GPUs (default 8)
# requires dp × tp × pp × ulysess_cp × ring_cp ≤ 16.
DTORCH_DTENSOR_IN_SAME_DEVICE=1 DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE=16 python3 python/dtorch/test/modules/test_llama.py
```
