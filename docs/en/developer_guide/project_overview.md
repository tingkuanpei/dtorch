# Project Overview

DTorch is a distributed deep learning API based on the **Single-Controller** and **Distributed Tensor** architecture, providing high-performance distributed computation for PyTorch. Users can scale a single-GPU PyTorch program to a multi-GPU distributed environment without modifying the code logic.

- **Language**: C++ (core engine), Python (framework layer and user interface)
- **Computation backend**: LibTorch (PyTorch C++ operator library)
- **Communication**: NCCL (GPU collective communication), ZMQ (inter-node message passing), gRPC (cluster RPC)
- **Build system**: CMake + scikit-build + nanobind

---

## Overall Architecture Layers

DTorch uses a layered architecture, forming a complete technology stack from the bottom-level infrastructure to the top-level API:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Python/C++ API Layer                           │
│           Python nn.Module / nanobind bridge / C++ Tensor API            │
├──────────────────────────────────────────────────────────────────────────┤
│                          C++ Core Engine Layer                           │
│              Operator system / LogicalGraph / Kernel engine              │
│  Blob memory mgmt / Stream scheduling / Distributed runtime / Comm libs  │
├────────────────────────┬─────────────────────────────────────────────────┤
│ C++ Common Base Layer  │            External Library Adapters            │
│ Log / assert / string  │    CUDA / NCCL / ZMQ / gRPC / Boost / Torch     │
│                        │          Sage Attention / Python utils          │
├────────────────────────┴─────────────────────────────────────────────────┤
│                       Infrastructure & Build Layer                       │
│         CMake build system / Third-party deps / Install scripts          │
└──────────────────────────────────────────────────────────────────────────┘
```

| Layer | Responsibility |
|---|---|
| **Python/C++ API Layer** | Python side: PyTorch-compatible nn.Module system, HuggingFace integration, CLI tools, tests; nanobind bridge; C++ side: public API wrappers for Tensor, Graph, DeviceMesh and functional operators |
| **C++ Core Engine Layer** | Operator definition and registration, Kernel execution, computation graph construction and execution, Runner execution scheduling, Blob memory management, Stream scheduling, distributed runtime, collective communication |
| **C++ Common Base Layer** | Logging system, argument parsing, debug assertions, string handling, subprocess management, environment variables, filesystem operations |
| **External Library Adapters Layer** | C++ adapter wrappers for third-party libraries: CUDA device management, NCCL communication, Boost serialization, ZMQ communication, gRPC communication, Torch utilities, Sage Attention acceleration |
| **Infrastructure & Build Layer** | CMake build system, third-party dependency management, install scripts, code formatting config, Python packaging |

---

## Directory Structure Overview

```
dtorch/
├── api/
│   ├── cpp/              # C++ public API (Tensor, DeviceMesh, functional)
│   └── python/           # nanobind Python bindings
├── common/               # common base library (logging, debug, string, filesystem)
├── core/
│   ├── blob/             # Blob physical memory containers
│   ├── communication/    # TensorStore, ThreadGroup communication abstractions
│   ├── distributed/      # MainNode / WorkerNode distributed node management
│   ├── graph/            # LogicalGraph / GraphConstructor / EagerGraphExecutor
│   ├── kernel/           # Kernel execution engine (TorchKernel etc.)
│   ├── kernel_stream/    # Stream scheduling (Cpu/CudaKernelStream)
│   ├── operand/          # Operand tensor metadata
│   ├── operators/        # Operator system
│   │   ├── standard/     #   standard operators (LibTorch backend)
│   │   ├── fused_compile/#   fused compilation operators
│   │   ├── system/       #   system operators
│   │   └── ...           #   operator base classes, factory, PlacementSignature
│   └── runner/           # Runner execution layer (NaiveRunner, PerDeviceThread/ProcessNodeRunner etc.)
├── external/
│   ├── boost/            # Boost serialization adapter
│   ├── cuda/             # CUDA device/stream/event management
│   ├── nccl/             # NCCL communication adapter
│   ├── python/           # Python utilities (type parsing etc.)
│   ├── rpc/              # gRPC service definitions and implementations
│   ├── sage_attn/        # Sage Attention acceleration adapter
│   ├── torch/            # LibTorch utilities
│   └── zmq/              # ZMQ messaging (RemoteRunnerPublisher/Subscriber/Pusher/Puller)
└── tests/                # C++ unit/integration tests

python/dtorch/
├── __init__.py           # package entry, API exports
├── nn/                   # nn.Module system
│   ├── module.py         #   DTorchModule base class
│   ├── linear.py         #   Linear / distributed variants
│   └── ...
├── applications/
│   └── huggingface/
│       ├── transformers/ #   T5 and other models
│       └── diffusers/    #   SD3 / Flux (UNet, Pipeline, Scheduler)
└── test/                 # Python test suite
    ├── operators/        #   operator tests (PyTorch baseline)
    ├── modules/          #   module tests
    └── ...
```

---

## Three Core Design Concepts

DTorch is built around three core design concepts:

- **Single-Client Single-Controller Multi-Worker** — asynchronous distributed execution model
- **Distributed Tensor** — native multi-dimensional distributed tensor abstraction
- **Eager Graph Architecture** — an execution engine combining the strengths of Eager Mode and Graph Mode

See the early chapters of [Design Decisions](design_decisions.md) for the design motivations of each concept, and [Key Concepts](key_concept.md) for detailed introductions.
