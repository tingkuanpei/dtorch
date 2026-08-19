# LogicalGraph Computation Graph Representation

DTorch's computation graph is built on three core abstractions: **LogicalGraph** (the computation graph container), **Operand** (a tensor metadata node) and **Operator** (a computation node). Operands and Operators form a DAG (directed acyclic graph) LogicalGraph, expressing the computation logic through its topology.

## 1. Architecture Overview

- **Operand**: a **data node** in the graph, holding a tensor's metadata (Shape, Stride, DataKind, DeviceMesh, Placements), not the actual data.
- **Operator**: a **computation node** in the graph, holding references to input/output Operands, encapsulating the metadata inference, distributed rules and actual computation logic of a single operator (such as `relu`, `add`, `matmul`).
- **LogicalGraph**: the DAG topology container of Operands and Operators, managing only node references.

## 2. Operand — the data node

**Source file**: `dtorch/core/operand.h`

`Operand` is the **data node** in the computation graph, representing the metadata of a tensor. It holds no actual data, only the tensor's descriptive information.

### Core members

```
Operand
├── Topology
│   ├── mProducerOp    — the Operator producing this Operand (exactly one)
│   └── mConsumerOps   — the list of Operators consuming this Operand (may be many)
│
├── Meta Info
│   ├── mShape         — the tensor's shape
│   ├── mStride        — the tensor's stride
│   └── mDataKind      — the data type (float32, float16, bfloat16, int64, etc.)
│
├── Distribute
    ├── mDeviceMesh    — the N-dimensional device mesh, describing the GPU topology
    └── mPlacementSeq  — the distributed placement sequence (Shard/Replicate/Partial)
```

### Topology relations

Each Operand is bidirectionally connected to Operators through `mProducerOp` and `mConsumerOps`:

- **ProducerOp**: points to the unique Operator that produces this Operand. The ProducerOp of an input Operand of the graph is `nullptr`.
- **ConsumerOps**: the list of all Operators consuming this Operand. The ConsumerOps of the graph's final output Operand is empty.

## 3. Operator — the computation node

**Source file**: `dtorch/core/operators/operator.h`

`Operator` is the **computation node** in the computation graph. Every operator the user calls (such as `relu`, `add`, `matmul`) corresponds to an Operator node in the graph, connecting input Operands to output Operands:

```
mInputOperands[0] ──┐
mInputOperands[1] ──┤
        ...         ├──► Operator ──► mOutputOperands[0]
mInputOperands[N] ──┘                  mOutputOperands[...]
```

`Operator` is a base class; every operator has a corresponding derived class (such as `ConvOp`, `LinearOp`, `ReduceOp`), using the template method pattern to implement its own behavior on top of the base class.

Each Operator derived class encapsulates:

- **Metadata inference** — infers the output Operands' metadata (Shape, DeviceMesh, Placements, etc.) from the input Operands' metadata; this process involves no actual data
- **Distributed rules** — declares the input/output Placements combinations the operator supports (PlacementSignature)

Each Operator also holds an **`OpParam`**, which describes all the parameters the operator needs — e.g., a convolution's `ConvParam` carries `kernelSize`, `pads`, `strides`, etc.

## 4. LogicalGraph — the DAG topology container

**Source file**: `dtorch/core/graph/logical_graph.h`

`LogicalGraph` is the top-level container of the computation graph, managing all nodes with two `unordered_map`s:

| Member | Type | Description |
|---|---|---|
| `mOperatorMap` | `unordered_map<const Operator*, shared_ptr<Operator>>` | operator map; key is a raw pointer, value is a smart pointer, supporting O(1) lookup |
| `mOperandMap` | `unordered_map<const Operand*, shared_ptr<Operand>>` | operand map, same as above |

### Key operations

| Method | Description |
|---|---|
| `AddOperator(op)` | adds an operator node to the graph |
| `AddOperand(operand)` | adds an operand node to the graph |
| `DeleteOperator(op)` | deletes an operator node (removes the reference only, does not update the graph topology) |
| `DeleteOperand(operand)` | deletes an operand node (removes the reference only, does not update the graph topology) |

`LogicalGraph` itself does **not maintain the graph's traversal order**; topological sorting is done by `GraphTraversalSequence`.

### Lifetime

In Eager mode, the `LogicalGraph` is held by `EagerGraphExecutor` (see `eager_graph_executor.h:97`). The graph **grows dynamically**: new nodes are added whenever the user calls an `api::cpp::functional` interface; each newly added node executes only once and is destroyed after execution completes.

## 5. Graph construction

### GraphConstructor

**Source file**: `dtorch/core/graph/graph_constructor.h`

`GraphConstructor` is the **bridge** between the Python API and the core engine. When the user calls an API such as `dtorch.functional.relu()`:

1. The Python layer calls the C++ API through nanobind
2. The API layer creates an `OpParam` and calls `GraphConstructor::AddOperator()`
3. `GraphConstructor` instantiates the corresponding `Operator` and executes `Infer()` to infer the metadata
4. The operator is sent to `EagerGraphExecutor` through `SendOperatorToExecutor()`

### GraphTraversalSequence

**Source file**: `dtorch/core/graph/graph_traversal_sequence.h`

The order in which `GraphConstructor` creates `Operators` is kept in `class GraphTraversalSequence`. `class GraphTraversalSequence` encapsulates an operator traversal sequence based on `std::list`, providing:
- O(1) lookup, insertion, deletion (through the `mNodeMap` auxiliary map)
- a bidirectional iterator interface
- maintenance of the execution order

## 6. Source file index

| File | Description |
|---|---|
| `dtorch/core/graph/logical_graph.h` `.cc` | `LogicalGraph` class — the DAG topology container |
| `dtorch/core/operand.h` `.cc` | `Operand` class — the tensor metadata data node |
| `dtorch/core/operators/operator.h` `.cc` | `Operator` base class — the computation node |
| `dtorch/core/graph/graph_constructor.h` | `GraphConstructor` class — the graph builder |
| `dtorch/core/graph/graph_traversal_sequence.h` | `GraphTraversalSequence` class — the topology traversal sequence |
