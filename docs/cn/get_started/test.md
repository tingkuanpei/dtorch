# 测试指南

本文档描述 DTorch 项目的三层测试体系：Python 单元测试、C++ 单元测试、应用测试。

---

## 1. Python 单元测试

测试框架采用 Python 标准库 `unittest`，所有算子测试以 PyTorch 为参考基准，验证 DTorch 输出与 PyTorch 一致。

### 目录结构

```
python/dtorch/test/
├── run_all_test.py         # 测试总入口，支持 scope 参数选择测试范围
├── test_tensor.py          # Tensor 基础 API 测试
├── test_graph.py           # Graph 图相关测试
├── test_distributed_tensor.py  # DTensor 分布式张量测试
├── test_util.py            # 测试公共工具（assert_tensor_allclose, gen_arg_list 等）
├── test_cluster.py         # 集群相关测试
├── operators/
│   ├── test_functioanl.py  # functional 算子 API 测试
│   ├── test_activation.py
│   └── ...
└── modules/                # 模型测试
    ├── test_llama.py
    ├── test_flux_1.py
    └── test_stable_diffusion_3.py
    └── ...
```

### 运行方式

**运行所有测试：**

```bash
# 默认运行 base + operators 测试
python3 python/dtorch/test/run_all_test.py

# 指定测试范围（scope 可叠加：base, operators, modules, all）
python3 python/dtorch/test/run_all_test.py base operators modules
python3 python/dtorch/test/run_all_test.py base+operators
python3 python/dtorch/test/run_all_test.py all
```

Scope 说明：

| Scope | 目录 |
|---|---|
| `base` | `python/dtorch/test/`（不含子目录） |
| `operators` | `python/dtorch/test/operators/` |
| `modules` | `python/dtorch/test/modules/` |
| `system_overhead` | `python/dtorch/test/system_overhead/` |
| `all` | 以上全部 |

Scope 支持 `+` 或 `,` 分隔组合，如 `base+operators`、`base,modules`。默认值为 `base operators`。

**运行单个测试文件：**

```bash
# 直接运行 Python 文件
python3 python/dtorch/test/operators/test_activation.py

# 或使用 pytest
pytest -s python/dtorch/test/operators/test_activation.py
```

**运行指定测试方法：**

```bash
# pytest 指定 TestClass.test_method
pytest -s python/dtorch/test/test_tensor.py::TestTensor::test_tensor_constructor_time

# 运行 DTorch vs PyTorch 一致性测试
python3 python/dtorch/test/operators/test_functioanl.py
```

## 2. C++ 单元测试

C++ 测试使用 **Google Test (gtest)** 框架，位于 `dtorch/tests/` 目录。

### 目录结构

```
dtorch/tests/
├── CMakeLists.txt          # 测试构建配置
├── test.h                  # gtest 头文件引入 + 编译器警告抑制
├── test_main.cc            # 测试入口 (InitGoogleTest + ArgumentParser)
├── test_graph.cc           # Graph / LogicalGraph 图构建与执行测试
├── test_remote_runner.cc   # RemoteRunner 远程执行测试
├── test_tensor_store.cc    # TensorStore 张量交换测试
└── ...

```

### 构建

C++ 测试随项目一起构建。请先完成[构建配置](how_to_build.md)：

```bash
# Editable install 会自动构建 C++ 测试
pip install -v --no-build-isolation -e .
```

### 运行方式

```bash
# 运行所有 C++ 测试
./build/test

# 运行特定测试（gtest filter）
./build/test --gtest_filter=SpecificTest.*

# 运行特定测试用例
./build/test --gtest_filter=LogicalGraphTest.BasicAdd
```

测试入口 (`test_main.cc`) 会先初始化 `ArgumentParser`，再调用 `RUN_ALL_TESTS()` 执行所有 Google Test 用例。

---

## 3. 应用测试

应用测试位于 `python/dtorch/applications/test/`，用于验证 HuggingFace Diffusers 模型在 DTorch 分布式执行下的输出正确性和性能。

### 目录结构

```
python/dtorch/applications/test/
├── __init__.py
└── test_model.py         # Flux / Stable Diffusion 3 集成测试
```

### 支持的模型

| 模型 | 命令行参数 |
|---|---|
| Stable Diffusion 3 Medium | `sd3`（默认） |
| FLUX.1-dev | `flux` |

### 运行方式

```bash
# 运行默认模型 (SD3)
python3 python/dtorch/applications/test/test_model.py

# 指定模型
python3 python/dtorch/applications/test/test_model.py sd3
python3 python/dtorch/applications/test/test_model.py flux

# 运行所有模型
python3 python/dtorch/applications/test/test_model.py all

# 启用输出校验（对比 DTorch 与 PyTorch 结果）
python3 python/dtorch/applications/test/test_model.py sd3 --check-output

# 指定迭代次数
python3 python/dtorch/applications/test/test_model.py sd3 --test-time 5

# 启用性能 benchmark
python3 python/dtorch/applications/test/test_model.py sd3 --benchmark

# 启用 torch.compile（参考管线）
python3 python/dtorch/applications/test/test_model.py sd3 --torch-compile

# 自定义 prompt
python3 python/dtorch/applications/test/test_model.py flux --prompt "A beautiful sunset over the ocean"

# 指定随机种子（保证可复现）
python3 python/dtorch/applications/test/test_model.py sd3 --generator-seed 42
```

### 测试流程

应用测试按以下流程执行：

1. **PyTorch 参考运行** — 使用原生 Diffusers Pipeline 生成参考图像
2. **DTorch 分布式运行** — 遍历 `test_configs` 中配置的各种并行策略组合（Data Parallel、Context Parallel、Ring Context Parallel 等）
3. **输出对比**（`--check-output`） — 通过 `TensorChecker.tensor_meanclose` 比较 DTorch 输出与 PyTorch 参考
4. **性能测量**（`--benchmark`） — 使用 `Benchmark` 工具测量并汇总各配置的推理耗时

### 测试配置说明

`test_model.py` 中的 `test_configs` 按模型定义，覆盖以下并行策略组合：

| 参数 | 说明 |
|---|---|
| `dp` | Data Parallel 并行度 |
| `ulysess_cp` | Ulysses Context Parallel 并行度 |
| `ring_cp` | Ring Context Parallel 并行度 |
| `first_block_cache` | 是否启用 First Block Cache（残差缓存优化） |
| `sage_attn` | 是否启用 Sage Attention 量化 |
| `fuse_kernel` | 是否启用 Kernel 融合（RoPE、SiLU-Linear、LayerNorm） |

---

## 快速参考

```bash
# Python 单元测试 — 全部
python3 python/dtorch/test/run_all_test.py all

# Python 单元测试 — 指定范围
python3 python/dtorch/test/run_all_test.py base operators

# Python 单元测试 — 单个文件
python3 python/dtorch/test/operators/test_matmul.py

# Python 单元测试 — 单个方法
pytest -s python/dtorch/test/test_tensor.py::TestTensor::test_tensor_constructor_time

# C++ 单元测试 — 全部
./build/test

# C++ 单元测试 — 过滤
./build/test --gtest_filter=LogicalGraphTest.*

# 应用测试 — SD3
python3 python/dtorch/applications/test/test_model.py sd3 --check-output

# 应用测试 — Flux
python3 python/dtorch/applications/test/test_model.py flux --check-output
```
