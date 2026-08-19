# Get Started

This page shows how to build and install DTorch, then verify distributed inference of the Llama model and the diffusion models (SD3 / FLUX).

---

## 1. Build and Install

**① Install system dependencies and third-party libraries** (Ubuntu 22.04, one-time):

```bash
apt install libboost-all-dev
script/download_third_party_lib.sh
script/install_zmq_ubuntu.sh
script/install_grpc_ubuntu.sh
```

**② Preinstall build tools and `torch`.**

```bash
pip install scikit-build-core torch==2.8.0 cmake==4.3.2
```

**③ Editable install.** This step automatically installs the remaining runtime dependencies (`diffusers`, `transformers`, `numpy`, `accelerate`, etc.):

```bash
pip install -v --no-build-isolation -e .
```

---

## 2. Llama Model Distributed Test

`python/dtorch/test/modules/test_llama.py` uses a single-GPU PyTorch model as reference (a small Llama with random weights, no weight download needed) to verify that DTorch output matches PyTorch under any combination of DP / TP / PP / CP. See [Llama parallel example](https://tingkuanpei.github.io/dtorch/cn/user_guide/llama_parallel/) for the parallel implementation details.

```bash
python3 python/dtorch/test/modules/test_llama.py
```

Without a multi-GPU cluster, you can enable [single-device distributed simulation](https://tingkuanpei.github.io/dtorch/cn/user_guide/python_api_overview/#6-彩蛋单卡模拟分布式) to run all strategy combinations on a single GPU:

```bash
# DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE sets the number of simulated GPUs (default 8)
DTORCH_DTENSOR_IN_SAME_DEVICE=1 DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE=16 python3 python/dtorch/test/modules/test_llama.py
```

---

## 3. Diffusion Model Test

### 3.1 Prepare Model Weights

The test script looks for models under `test_data/` in the repository root by default; override with the `DTORCH_TEST_DATA_DIR` environment variable. Create a subdirectory named after the model:

| Model | Argument | Subdirectory name |
|---|---|---|
| Stable Diffusion 3 | `sd3` | `stable-diffusion-3-medium-diffusers` |
| FLUX.1-dev | `flux` | `FLUX.1-dev` |

### 3.2 Run the Application Test

The script first runs the PyTorch reference pipeline, then executes DTorch distributed inference under the predefined parallel strategy combinations.

```bash
# Default model (SD3); options: sd3 / flux / all
python3 python/dtorch/applications/test/test_model.py sd3
```

Common options: `--check-output` (compare against the reference output), `--benchmark`, `--no-torch` (skip the torch reference run), `--prompt`, `--generator-seed`, `--test-count`.

On success, `.jpg` images are generated in the current directory and `All tests for '<model>' passed.` is printed. See the [testing guide](https://tingkuanpei.github.io/dtorch/cn/get_started/test/) for more details.
