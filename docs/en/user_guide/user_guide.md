# User Guide

This guide gets you started with DTorch in two steps: first understand the core concepts of the distributed tensor (DTensor), then learn how to write distributed programs with the DTorch Python API.

## Recommended Reading Order

1. **[DTensor Overview](distributed_tensor_overview.md)** — Learn how `DeviceMesh` and `Placements` describe the distribution of a tensor across devices. This is the foundation of all of DTorch's distributed capabilities; read this first.
2. **[Python API Overview](python_api_overview.md)** — Starting from single-GPU PyTorch code, just declare `device_mesh` and `placements` to scale up to a multi-GPU distributed program.
3. **[Module Parallel](module_parallel.md)** — How `Linear` and `nn.Module` combine to implement DP / TP / CP.
4. **[Llama Parallel Example](llama_parallel.md)** — Using the Llama model as an example, showing the complete implementation and testing of DP + TP + PP + CP.
