# Python Kernel：在 C++ Kernel 中调用 Python 代码

DTorch 的 Kernel 默认通过 `Operator::Compute()` 调用 LibTorch C++ 算子。但在某些场景下（如使用 Python 生态的优化库），需要在 C++ Kernel 的执行路径中调用 Python 代码。本文档以 **Sage Attention**（`SdpaOp`）和 **Fused Compile**（`dtorch.compiled_op`）两类 Python Kernel 场景为例，说明完整机制。

## 1. 使用场景

当第三方加速库仅提供 Python 接口（如 Sage Attention、Flash Attention v2 Python binding 等），或用户希望在 Kernel 中执行自定义的 Python 编译逻辑时，需要在 C++ Kernel 中嵌入 Python 调用。

DTorch 中 Python Kernel 的两种典型使用场景：

| 场景 | Python 模块 | 使用位置 | 说明 |
|---|---|---|---|
| 第三方加速库 | `sageattention` | `SdpaOp::Compute()` | 根据参数选择 LibTorch 原生或 Python Sage Attention 路径 |
| 编译融合算子 | `dtorch.compiled_op` | `fused_compile/*.cc` | 在 Compute 中调用 Python 侧的 torch.compile / 自定义融合实现 |

**源文件**:
- `dtorch/core/operators/standard/scaled_dot_product_attention_op.cc` — SdpaOp 示例
- `dtorch/core/operators/fused_compile/layer_norm_mul_add.cc` — 编译融合算子示例
- `dtorch/core/operators/fused_compile/silu_linear_chunk.cc` — 编译融合算子示例
- `dtorch/core/operators/fused_compile/apply_rotary_emb_op.cc` — 编译融合算子示例
- `dtorch/external/sage_attn/sage_attn_adapter.cc` — Sage Attention C++ 适配器

## 2. 核心调用模式

```
Operator::Compute()
  │
  ├─ (默认路径) torch::native_op(...)            — LibTorch C++ 算子
  │
  └─ (Python Kernel 路径) 调用 Python 函数        — 第三方 Python 加速库 / 编译融合算子
       │
       ├─ GilScopedAcquire                     — 获取 GIL
       ├─ PythonCodeCudaStreamGuard            — 保护 CUDA Stream
       ├─ nb::module_::import_("xxx")      — 导入 Python 模块
       ├─ 调用 Python 函数                      — 执行计算
       └─ NanobindUtil 类型转换                 — Tensor ↔ nb::object
```

每个 Python Kernel 调用都遵循相同的五步模式：

```cpp
// 1. 获取 GIL（必须在调用任何 Python API 前）
auto scopedAcquire = external::python::GilScopedAcquire();
// 2. 保存并保护 CUDA Stream（必须在获取 GIL 之后）
auto streamGuard = external::torch::PythonCodeCudaStreamGuard();
// 3. 动态导入 Python 模块
nb::module_ compileKernelModule = nb::module_::import_("dtorch.compiled_op");
// 4. 调用 Python 函数（使用 NanobindUtil 转换参数）
nb::object funcOut = compileKernelModule.attr("layer_norm_mul_add")(
    NanobindUtil::ToObject(emb), NanobindUtil::ToObject(scale), ...);
// 5. 将返回值转换为 C++ Tensor
auto outputs = NanobindUtil::ToTensorArray(funcOut);
// (streamGuard 和 scopedAcquire 按构造逆序自动析构)
```

## 3. 示例一：Sage Attention 适配器

`SdpaOp::Compute()` 根据 `sageAttentionType` 参数选择 LibTorch 原生或 Sage Attention Python 路径。`SageAttnAdapter` 封装了 GIL 管理、CUDA Stream 保护和 nanobind 类型转换，Operator 层只需像调用普通 C++ 函数一样调用适配器接口。

**源文件**: `dtorch/core/operators/standard/scaled_dot_product_attention_op.cc`、`dtorch/external/sage_attn/sage_attn_adapter.cc`

## 4. 示例二：Fused Compile 编译融合算子

与 Sage Attention 不同，融合算子直接在 `Compute()` 中内联 Python 调用，未通过外部适配器封装：

```cpp
auto scopedAcquire = external::python::GilScopedAcquire();
auto streamGuard = external::torch::PythonCodeCudaStreamGuard();
nb::module_ compileKernelModule = nb::module_::import_("dtorch.compiled_op");
nb::object funcOut = compileKernelModule.attr("layer_norm_mul_add")(...);
auto outputs = NanobindUtil::ToTensorArray(funcOut);
```

**源文件**: `dtorch/core/operators/fused_compile/layer_norm_mul_add.cc`、`silu_linear_chunk.cc`、`apply_rotary_emb_op.cc`

## 5. C++ 调用 Python 的基础设施

### 5.1 GIL 管理 (`GilScopedAcquire` / `GilScopedRelease`)

**源文件**: `dtorch/external/python/python_gil.h`

在多线程 C++ 环境中调用 Python 代码时，必须先获取 Python 全局解释器锁 (GIL)：

```cpp
class GilScopedAcquire {
public:
    GilScopedAcquire() : state(PyGILState_Ensure()) {}
    ~GilScopedAcquire() { PyGILState_Release(state); }
private:
    PyGILState_STATE state;
};

class GilScopedRelease {
public:
    GilScopedRelease() : state(nullptr) {
        DAlwaysAssert(PyGILState_Check());
        state = PyEval_SaveThread();
    }
    ~GilScopedRelease() { PyEval_RestoreThread(state); }
private:
    PyThreadState *state;
};
```

- **`GilScopedAcquire`**: RAII 封装，构造时调用 `PyGILState_Ensure()` 获取 GIL，析构时调用 `PyGILState_Release()` 释放。用于进入 Python 交互前。
- **`GilScopedRelease`**: RAII 封装，构造时调用 `PyEval_SaveThread()` 释放 GIL，析构时调用 `PyEval_RestoreThread()` 恢复。用于在持有 GIL 但需要让其他线程执行 Python 代码时暂时释放。

> **重要**：PyTorch C++ Tensor 内部持有 `PyObjectSlot`，其析构函数需要获取 GIL。因此在多线程环境（DTorch 的 KernelStream 每个运行在独立线程上）中，释放 GIL 持有者的 C++ Tensor 时必须确保 GIL 已获取，否则可能导致死锁。

### 5.2 CUDA Stream 保护 (`PythonCodeCudaStreamGuard`)

**源文件**: `dtorch/external/torch/torch_stream_guard.h` / `.cc`

Python 代码（如 `sageattention` 库或 `torch.cuda` API）在执行 CUDA 操作时可能切换当前 CUDA Stream。`PythonCodeCudaStreamGuard` 在构造时：

1. 从 Python 侧获取当前 CUDA Stream（`torch.cuda.current_stream()`），保存为 `originalPythonCudaStream`
2. 从 C++ 侧获取当前 CUDA Stream（`at::cuda::getCurrentCUDAStream()`）
3. 将 Python 侧的 CUDA Stream 设置为 C++ 侧的当前流（通过 `torch.cuda.set_stream()` 和自定义 `ExternalStream`），使 Python 代码的输出写入 C++ KernelStream 的正确 CUDA Stream 中

析构时恢复 `originalPythonCudaStream`，确保后续 Python 调用不受影响。

```cpp
// 典型使用：GilScopedAcquire 先于 streamGuard 构造，后于 streamGuard 析构
auto scopedAcquire = GilScopedAcquire();                    // 1. 获取 GIL
auto streamGuard = external::torch::PythonCodeCudaStreamGuard(); // 2. 对齐 C++/Python CUDA Stream
// ... Python 调用 ...
// streamGuard 析构：恢复原始 Python CUDA Stream
// scopedAcquire 析构：释放 GIL
```

### 5.3 类型转换 (`NanobindUtil`)

**源文件**: `dtorch/external/python/nanobind_util.h`

`NanobindUtil` 提供 `torch::Tensor` 与 `nb::object` 之间的高效零拷贝转换：

| 方法 | 方向 | 说明 |
|---|---|---|
| `ToObject(tensor)` | C++ → Python | 将 `torch::Tensor` 包装为 nanobind object，调用 `THPVariable_Wrap` 实现零拷贝 |
| `ToTensor(obj)` | Python → C++ | 将 nanobind object 或 PyObject* 转换为 `torch::Tensor` |
| `ToTensorArray(obj)` | Python → C++ | 自动识别单个 Tensor 或 Tuple/List of Tensor，统一转换为 `std::vector<torch::Tensor>` |

`ToTensorArray` 内部自动区分单返回值和多返回值：

```cpp
static std::vector<torch::Tensor> ToTensorArray(PyObject* obj) {
    std::vector<torch::Tensor> result;
    if (!PyCheckIsTupleOrList(obj)) {
        result.push_back(ToTensor(obj));  // 单个 Tensor
    } else {
        for (auto it : PyUnpackTupleOrList(obj)) {
            result.push_back(ToTensor(it));  // Tuple/List 中的每个 Tensor
        }
    }
    return result;
}
```

## 6. PerDeviceProcessNodeRunner：多进程解决 GIL 瓶颈

Python GIL（见第 5.1 节）是全局互斥锁，多 GPU 线程共享同一进程的 GIL，导致 Python Kernel 无法真正并行。`PerDeviceProcessNodeRunner` 通过**进程隔离**解决此问题：每张 GPU 运行在独立的子进程中，拥有独立的 Python 解释器和 GIL，进程间通过 ZMQ IPC 通信。

**源文件**: `dtorch/core/runner/per_device_process_node_runner.h` / `.cc`、`dtorch/core/runner/process_remote_runner.h` / `.cc` |

## 7. Python Kernel 在完整执行流程中的位置

Python Kernel 调用发生在 `Kernel::Run()` → `Compute()` 阶段，与普通 Kernel 的执行流程完全一致。关键区别仅在 `Compute()` 内部走 Python 路径而非 LibTorch C++ 路径：

```
Kernel::Run()
  │
  ├─ 1. KernelHook::BeforeCompute()
  ├─ 2. CUDA Stream 校验
  ├─ 3. 准备输入                             — 从 Blob 提取 torch::Tensor
  ├─ 4. Compute(inputs, outputs)
  │    │
  │    └─ mOp->Compute(torch_inputs, torch_outputs)
  │         │
  │         ├─ [默认 C++ 路径]
  │         │   torch::scaled_dot_product_attention(inputs...)       ← LibTorch C++ 算子
  │         │
  │         └─ [Python Kernel 路径]
  │              external::sage::SageAttnAdapter(inputs...)           ← C++ 适配器
  │              │
  │              ├─ GilScopedAcquire()                                ← GIL 管理
  │              ├─ PythonCodeCudaStreamGuard()                       ← CUDA Stream 保护
  │              ├─ nb::module_::import_("sageattention")         ← 导入 Python 模块
  │              ├─ module.attr("sageattn")(ToObject(q), ToObject(k), ...)  ← 调用 Python 函数
  │              └─ NanobindUtil::ToTensorArray(result)               ← 结果转换
  │
  ├─ 5. 验证输出
  ├─ 6. 写回 Blob                             — Blob::SetTensor()
  └─ 7. KernelHook::AfterCompute()
```

## 8. 关键注意事项

### 8.1 GIL 死锁风险

在多线程 C++ 环境中（DTorch 的 KernelStream 每个运行在独立线程上），调用 Python 代码时必须正确管理 GIL：

- **调用 Python 前**：必须 `GilScopedAcquire`（或等效的 `PyGILState_Ensure()`）获取 GIL。
- **Python 调用结束后**：应及时释放 GIL（`GilScopedAcquire` 析构自动释放），避免阻塞其他需要 GIL 的线程。
- **释放 C++ Tensor 时**：注意 PyTorch Tensor 的 `PyObjectSlot` 析构可能需要 GIL。如果 Tensor 在释放 GIL 之后析构（如作为函数返回值持有），需确保在持有 GIL 的作用域内发生析构。

### 8.2 CUDA Stream 一致性

Python 代码中可能切换 CUDA Stream（如 `torch.cuda.set_stream()`），导致返回 C++ 后 CUDA Stream 不一致。解决方案：

- 使用 `PythonCodeCudaStreamGuard` 在调用 Python 前将 Python 侧 Stream 对齐到 C++ 侧的当前 CUDA Stream，确保 Python Kernel 的输出写入正确的 Stream。
- 该 Guard 应紧随 `GilScopedAcquire` 之后创建、在释放 GIL 之前析构。

### 8.3 Python 初始化

在 C++ 中内嵌 Python 解释器前，必须先调用 `nanobind::initialize_interpreter()`。DTorch 在启动时已完成此初始化（通过 `nanobind/nanobind.h`），因此 Operator 的 `Compute()` 中可以直接使用 nanobind API。

### 8.4 性能考虑

- **模块导入开销**：每次调用 `nb::module_::import_("xxx")` 会有字典查找开销。对于高频调用场景，可考虑在首次调用时缓存 `nb::module_` 对象的引用（如 `sage_attn_adapter.cc` 中每次函数调用都重新 import，对于批量调用可能产生可测量的开销）。
- **GIL 争用**：Python GIL 是全局互斥锁，同一时刻只有一个线程能执行 Python 字节码。对于多 GPU 并行且每个 Kernel 都走 Python 路径的场景，GIL 可能成为瓶颈。因此使用 Python Kernel 时，需要开启 PerDeviceProcessNodeRunner，每一个 GPU 具有独立的进程的模式。

- **零拷贝转换**：`NanobindUtil::ToObject()` 使用 `THPVariable_Wrap` 实现零拷贝，不会复制 Tensor 数据，仅增加 Python 引用计数。

## 9. 源文件索引

| 组件 | 头文件 | 实现文件 |
|---|---|---|
| GilScopedAcquire / GilScopedRelease | `dtorch/external/python/python_gil.h` | — |
| NanobindUtil | `dtorch/external/python/nanobind_util.h` | — |
| PythonCodeCudaStreamGuard | `dtorch/external/torch/torch_stream_guard.h` | `dtorch/external/torch/torch_stream_guard.cc` |
| SageAttnAdapter | `dtorch/external/sage_attn/sage_attn_adapter.h` | `dtorch/external/sage_attn/sage_attn_adapter.cc` |
| SdpaOp | `dtorch/core/operators/standard/scaled_dot_product_attention_op.h` | `dtorch/core/operators/standard/scaled_dot_product_attention_op.cc` |
| Fused Compile Ops | `dtorch/core/operators/fused_compile/*.h` | `dtorch/core/operators/fused_compile/*.cc` |

## 10. 相关文档

- [Kernel 运行时](kernel_runtime.md) — Blob、Kernel、KernelStream 执行映射
- [Operator 算子体系](../operator/operator.md) — Operator 基类、模板方法模式与完整生命周期
- [设计理念](../key_concept.md) — Eager Graph Architecture 三大核心设计
