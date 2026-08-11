# 排查 DTorch 与 PyTorch 模型输出不一致

当 DTorch 模型与 PyTorch 模型的输出结果不一致时，需要系统性地定位根因。本文档介绍一套从粗到细的排查方法论。

---

## 1. 理论依据

DTorch 的核心计算后端是 **LibTorch**（PyTorch 的 C++ 库），而用户使用的 PyTorch 通过 Python 接口层最终调用的也是 **同一套 LibTorch C++ 接口**。

```
DTorch:   Python API → nanobind → C++ Operator → LibTorch (torch::*)
PyTorch:  Python API → torch.nn → C++ ATen   → LibTorch (at::*)
```

因此，当满足以下条件时，DTorch 和 PyTorch **运行的是完全相同的 CUDA kernel**：

- 输入 Tensor 的值、Shape、dtype、device 完全一致
- 算子语义和参数完全一致
- 无量化、无分布式、无 cache 等干扰因素

在这种情况下，两者的输出应当**逐位完全一致**，可以使用 `torch.equal()` 进行精确对齐。`python/dtorch/test/operators/` 目录下的所有单机单卡算子测试均通过 `assert_tensor_equal` 验证了这一一致性。

---

## 2. 不适用的情况

以下场景中，DTorch 与 PyTorch 的输出**不能**期望逐位完全一致：

| 场景 | 原因 |
|---|---|
| **模型实现不一致** | DTorch 模型与 PyTorch 模型的结构、参数、算子不同，运行的 kernel 不同，输出必然不同 |
| **启用量化** | 量化推理引入了近似计算（如 FP16、INT8），精度损失导致输出不完全一致 |
| **启用 Cache** | First Block Cache 等优化缓存了中间计算结果，跳过了部分计算，可能引入微小差异 |
| **分布式计算** | `Shard` 后的浮点运算不满足交换律和结合律（如 `all_reduce(sum(x_i))` 的求和顺序不确定），导致多卡结果与单卡参考不完全一致 |
| **随机性算子** | dropout、sampling 等算子本身具有随机性，每次运行结果不同 |

对于上述场景，应使用 `check_allclose` / `assert_tensor_allclose` 进行近似比较，而非 `check_equal`。

---

## 3. 排查工具：TensorChecker

`python/dtorch/util/tensor_checker.py` 提供了 `TensorChecker` 类，用于对比 DTorch 与 PyTorch 的中间/最终输出。

### 3.1 核心 API

| 方法 | 用途 |
|---|---|
| `tensor_checker.register_tensor(tag, tensor)` | 注册一个待比较的 tensor。同一 tag 下的 tensor 会被分组比较 |
| `tensor_checker.check_equal()` | 对所有已注册的 tensor 执行 `torch.equal` 精确比较 |
| `tensor_checker.check_allclose(rtol, atol)` | 对所有已注册的 tensor 执行 `torch.allclose` 近似比较 |
| `tensor_checker.clear()` | 清空已注册的 tensor，准备下一轮比较 |

### 3.2 自动注册：`module_register_tensor_checker`

```python
from dtorch.util.tensor_checker import tensor_checker, module_register_tensor_checker

# 自动注册两个模型所有同名 submodule 的 parameters、buffers、inputs、outputs
module_register_tensor_checker(torch_model, tensor_checker)
module_register_tensor_checker(dtorch_model, tensor_checker)

# 运行一次 forward，所有中间 tensor 自动注册
torch_output = torch_module(torch_input)
dtorch_output = dtorch_module(dtorch_input)

# 比较
tensor_checker.check_equal()
```

**注意**：`module_register_tensor_checker` 要求两个模型的 `named_modules()` 返回的 submodule 名称**完全一致**，否则无法匹配。

### 3.3 Dump 模式

当 tensor 不一致时，可通过 `dump_mode` 将差异 tensor 保存为 `.npy` 文件，便于离线分析：

```python
from dtorch.util.tensor_checker import DumpMode

tensor_checker.check_equal(dump_mode=DumpMode.NOT_EQUAL, dump_path="/tmp/debug")
```

| DumpMode | 行为 |
|---|---|
| `NOT` | 不保存（默认） |
| `ALL` | 保存所有 tensor |
| `NOT_EQUAL` | 仅保存不一致的 tensor |
| `FIRST_NOT_EQUAL` | 仅保存第一个不一致的 tensor |

---

## 4. 排查步骤

### Step 1：粗粒度定位 — 找到不一致的 Module

首先使用 `module_register_tensor_checker` 在 Module 级别定位问题：

```python
from dtorch.util.tensor_checker import tensor_checker, module_register_tensor_checker

# 1. 加载 DTorch 和 PyTorch 模型（确保两者的权重完全一致）
torch_model = load_torch_model()
dtorch_model = load_dtorch_model()

# 2. 注册所有 submodule 的 tensor checker
module_register_tensor_checker(torch_model, tensor_checker)
module_register_tensor_checker(dtorch_model, tensor_checker)

# 3. 运行一次 forward
torch_model.eval()
dtorch_model.eval()
with torch.no_grad():
    torch_out = torch_model(torch_input)
    dtorch_out = dtorch_model(dtorch_input)

# 4. 比较所有中间结果
tensor_checker.check_equal()
```

输出示例：

```
TensorChecker: 2 tensors equal in checker_conv_in
TensorChecker: 2 tensors equal in checker_conv_in_input_0
TensorChecker: 2 tensors not equal in checker_downsample.1
TensorChecker: 2 tensors equal in checker_conv_out
...
```

从输出中定位到第一个出现不一致的 module（如上例中 `downsample.1`），它就是问题的起点。

### Step 2：细粒度定位 — 找到不一致的 Operator

确定是哪个 module 后，在该 module 内部手动注册单个 tensor 以进一步缩小范围：

```python
# 假设问题出在某个自定义 module 的 forward 中
class MyModule(torch.nn.Module):
    def forward(self, x):
        # PyTorch 版本
        x1 = torch.nn.functional.silu(x)            # ← 注册 x1
        x2 = x1 * 0.5                                # ← 注册 x2
        x3 = torch.nn.functional.layer_norm(x2, ...) # ← 注册 x3
        return x3

# 在 DTorch 对应的 forward 中同样注册
# dtorch 版本中：
#   tensor_checker.register_tensor("my_module.x1", x1)
#   tensor_checker.register_tensor("my_module.x2", x2)
#   tensor_checker.register_tensor("my_module.x3", x3)
```

**排查策略**：
- 在 module 的 forward 中，从输入到输出逐步注册中间 tensor
- 对每个关键算子（activation、normalization、linear、attention 等）的输入和输出分别注册
- 运行后比较，找到**第一个输出不一致的算子**

```python
# 注册算子输入
tensor_checker.register_tensor("my_module.silu_input", x)
# 注册算子输出
tensor_checker.register_tensor("my_module.silu_output", x1)
```

每次缩小范围后 `check_equal()`，直到定位到具体的算子。

### Step 3：根据根因分类处理

#### Case A：DTorch 算子计算逻辑与 PyTorch 不一致

如果两个版本的算子实现逻辑不同（例如 DTorch 使用了不同的公式、参数，或某个算子尚未实现而使用了近似替代），则应**让 DTorch 算子向 PyTorch 对齐**。

对应的单元测试应添加到 `python/dtorch/test/operators/` 目录中，以 PyTorch 为参考基准。

#### Case B：DTorch 算子逻辑正确但结果仍有差异

如果 DTorch 算子的实现逻辑与 PyTorch 完全一致（即调用的都是同一个 LibTorch C++ 函数），但输出仍然不一致，可能是 DTorch 在以下环节存在 bug：

- Shape 推断错误（`InferOutputMetaInfo`）
- 输入预处理或类型转换环节出错
- 内存布局（stride、contiguous）处理不当

此时需要深入到 DTorch 的 C++ 算子实现中排查，参考 [How To Add Operator](../operator/how_to_add_operator.md) 了解算子内部机制。

#### Case C：DTorch 尚未实现某个算子

如果某个算子 DTorch 尚未支持，则报错信息会明确指出。此时需要按照 [How To Add Operator](../operator/how_to_add_operator.md) 文档为该算子增加支持。

---

## 5. 总结

排查 DTorch 与 PyTorch 输出不一致的完整流程：

```
1. module_register_tensor_checker  →  定位到有问题的 Module
2. tensor_checker.register_tensor  →  定位到有问题的 Operator
3. 分析根因：
   ├── 计算逻辑不一致 → 向 PyTorch 对齐
   ├── DTorch 实现有 bug → 深入 C++ 算子排查（参考 how_to_add_operator.md）
   └── 算子未实现 → 新增算子（参考 how_to_add_operator.md）
4. 修复后重新验证，逐个对齐所有差异
```

核心原则：**DTorch 的每个算子都应能以 `torch.equal` 精度与 PyTorch 对齐**（排除第 2 章列出的不适用场景）。通过上述方法逐步缩小范围、逐个算子对齐，最终 DTorch 与 PyTorch 的模型输出就能达到完全一致。
