# DTorch

![Status](https://img.shields.io/badge/status-beta-orange)
![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599c)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-ee4c2c)
![Platform](https://img.shields.io/badge/platform-Linux%20%2F%20x86__64-lightgrey)

**DTorch** is a distributed deep learning API for PyTorch, built on the **Single-Controller** and **Distributed Tensor (DTensor)** architectures. It lets you scale a single-GPU PyTorch program to a multi-GPU, multi-node cluster **without changing your computation logic** — you only declare a `DeviceMesh` and `Placements`, and DTorch takes care of resource management, scheduling, and communication.

---

## Highlights

- **PyTorch-compatible API.** On a single device, DTorch behaves identically to PyTorch — tensors, operators, modules, and dtypes map one-to-one. Existing model code needs only import substitutions to run.
- **Native DTensor.** Describe N-dimensional parallelism with `DeviceMesh` + `Placements` (`Shard` / `Replicate` / `Partial`). All operators infer the output's placement automatically — no manual sharding or gathers.
- **Declarative redistribution.** Change a tensor's layout with `tensor.redistribute(...)`. DTorch inserts the right collective (AllReduce, AllGather, ReduceScatter, …) and overlaps it with computation — **no `ProcessGroup` or explicit collectives in user code.**
- **Unified parallel strategies.** Data Parallel, Tensor Parallel, Context Parallel (Ulysses & Ring), Pipeline Parallel, Expert Parallel, and ZeRO — all expressible in the same code, and **freely composable**.
- **Eager Graph Architecture.** An eager user interface backed by an asynchronous, graph-aware execution engine that enables compute/communication overlap, operator fusion, redundancy elimination, and memory reuse.
- **Single-GPU distributed simulation.** Develop and debug a multi-way parallel program on a **single** GPU before deploying to a real cluster — DTorch's logical `DeviceMesh` simulates the sharding and collectives for you.

---

## How It Works

DTorch is organized around three core design concepts.

### 1. Single-Client Single-Controller Multi-Worker

An asynchronous execution model with three cooperating roles:

```
┌──────────────┐      async messages      ┌─────────────────┐     kernel queue     ┌───────────┐
│  Client      │ ───────────────────────> │   Controller    │ ───────────────────> │  Worker   │
│  (Python)    │                          │  (C++ MainNode) │                      │ (C++ Thr) │
│              │ <── sync only on value ── │                 │ <─ sync only on val ─│           │
└──────────────┘                          └─────────────────┘                      └───────────┘
```

The **Client** (a single-threaded Python process) creates DTensor symbols and operators, serialized and sent asynchronously. The **Controller** (one `MainNode` per cluster, one `WorkerNode` per machine) builds the logical graph and manages all compute resources. Each **Worker** is a C++ thread bound to a CUDA stream that executes kernels in order. The pipeline is fully asynchronous and **synchronizes only when a tensor value is read**.

### 2. Distributed Tensor (DTensor)

`DeviceMesh` describes the GPU topology as an N-dimensional, named grid; `Placement` describes how a tensor is distributed along each mesh dimension. Three placements cover every mainstream parallel scheme:

| Placement | Meaning |
|---|---|
| `Shard(dim)` | Split the tensor along axis `dim` across devices |
| `Replicate()` | Full copy on every device |
| `Partial()` | Each device holds a partial result (reduced later) |

### 3. Eager Graph Architecture

A four-layer engine — *graph representation → kernel runtime → Eager Graph engine → collective communication* — that exposes a simple eager API while gaining graph-level optimizations. The Client emits incremental sub-graphs asynchronously; the Controller rewrites and executes them with a three-level concurrency model (between graphs, between operators, and within operators).

---

## Documentation

Full documentation lives under [`docs/`](docs/).

| Topic | Document |
|---|---|
| Architecture overview | [docs/developer_guide/architecture.md](docs/developer_guide/architecture.md) |
| Design concepts | [docs/developer_guide/design_concept.md](docs/developer_guide/design_concept.md) |
| Python API & DTensor | [docs/user_guide/python_api.md](docs/user_guide/python_api.md) |
| Distributed Tensor | [docs/user_guide/distributed_tensor.md](docs/user_guide/distributed_tensor.md) |
| Single-Controller architecture | [docs/developer_guide/single_controller.md](docs/developer_guide/single_controller.md) |
| Eager Graph architecture | [docs/developer_guide/eager_graph_architecture/](docs/developer_guide/eager_graph_architecture/) |
| Diffusion model applications | [docs/user_guide/applications.md](docs/user_guide/applications.md) |
| Build guide | [docs/get_started/how_to_build.md](docs/get_started/how_to_build.md) |
| Test guide | [docs/get_started/test.md](docs/get_started/test.md) |
