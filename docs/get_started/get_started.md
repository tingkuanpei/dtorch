# 快速开始

本页介绍如何编译安装 DTorch，并验证 Llama 模型与扩散模型（SD3 / FLUX）的分布式推理。

---

## 1. 编译安装

**① 安装系统依赖与第三方库**（Ubuntu 22.04，一次性）：

```bash
apt install libboost-all-dev
script/download_third_party_lib.sh
script/install_zmq_ubuntu.sh
script/install_grpc_ubuntu.sh
```

**② 预装构建工具与 `torch`**。

```bash
pip install scikit-build-core torch==2.8.0 cmake==4.3.2
```

**③ 可编辑安装**。此步会自动装入其余运行依赖（`diffusers`、`transformers`、`numpy`、`accelerate` 等）：

```bash
pip install -v --no-build-isolation -e .
```

---

## 2. Llama 模型分布式测试

`python/dtorch/test/modules/test_llama.py` 以 PyTorch 单卡模型为参考（随机权重的小型 Llama，无需下载权重），验证 DP / TP / PP / CP 任意组合下 DTorch 输出与 PyTorch 一致。并行实现细节见 [Llama 并行示例](../user_guide/llama_parallel.md)。

```bash
python3 python/dtorch/test/modules/test_llama.py
```

没有多卡集群时，可开启[单卡模拟分布式](../user_guide/python_api_overview.md#6-彩蛋单卡模拟分布式)，在单张 GPU 上运行全部策略组合：

```bash
# DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE 指定模拟的 GPU 数量（默认 8）
DTORCH_DTENSOR_IN_SAME_DEVICE=1 DTORCH_NUM_GPU_WHEN_ENABLE_DTENSOR_IN_SAME_DEVICE=16 python3 python/dtorch/test/modules/test_llama.py
```

---

## 3. Diffusion 模型测试

### 3.1 准备模型权重

测试脚本默认在仓库根目录的 `test_data/` 下查找模型，可用环境变量 `DTORCH_TEST_DATA_DIR` 覆盖。按模型名建立子目录：

| 模型 | 参数 | 子目录名 |
|---|---|---|
| Stable Diffusion 3 | `sd3` | `stable-diffusion-3-medium-diffusers` |
| FLUX.1-dev | `flux` | `FLUX.1-dev` |

### 3.2 运行应用测试

脚本先运行 PyTorch 参考管线，再对预定义的并行策略组合执行 DTorch 分布式推理。

```bash
# 默认模型 (SD3)，可选 sd3 / flux / all
python3 python/dtorch/applications/test/test_model.py sd3
```

常用选项：`--check-output`（对比参考输出）、`--benchmark`、`--no-torch`（跳过 torch 参考运行）、`--prompt`、`--generator-seed`、`--test-count`。

运行成功会在当前目录生成 `.jpg` 图像并打印 `All tests for '<model>' passed.`。更多测试说明见 [测试指南](test.md)。
