# Performance

## 1. 显存占用

受益于 Client、Controller 和 Worker 异步计算的特性，DTorch 在计算过程中的中间 Tensor 可以及时释放，因此显存占用低于 PyTorch。StableDiffusion3 模型的实测数据如下表所示：

<figure markdown>
  ||PyTorch|DTorch|
  |-|-|-|
  |峰值占用|18.131GB|17.663GB<span style="color: #4caf50; font-weight: bold;">(-2.58%)</span>|

  <figcaption>表 1：StableDiffusion3 单机单卡 1024*1024 4step 峰值显存</figcaption>
</figure>

下面的代码片段解释了这一差异的原因：变量 a、b、c 是计算过程中的中间变量，由于没有及时释放，会推高 PyTorch 的峰值显存。而 DTorch 采用异步计算，构图的 Python 线程提前执行完 `func()` 并返回，C++ 执行引擎由此得知中间变量 a、b、c 不再被持有，从而可以及时释放其显存。

```python
def func():
    a = torch.rand(...)
    b = operator0(a)
    c = operator0(b)
    d = operator0(c)
    return d
```

## 2. 性能

### 2.1 算子层面

理论上，Single-Controller 的调度开销大于 Multi-Controller，但得益于 DTorch 在 C++ 层的深度优化，DTorch 在小算子上的开销仅比 PyTorch 高 37%，而在大算子上两者耗时基本一致。

下图是不同 shape 的 Tensor 在 CPU 和 GPU 上执行加法的耗时。即使是 `Shape=(1,)` 的 Tensor 的加法，DTorch 也仅比 PyTorch 高 37%；随着 Tensor 变大，两者耗时基本一致。在计算密集的 SDPA 算子上也能得到相同的结论。

<figure markdown>
  ![不同 shape 的 Tensor add 算子的耗时](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/tensor_add_performance.png)
  <figcaption>图 1：不同 shape 的 Tensor add 算子的耗时（NVIDIA 4090）</figcaption>
</figure>

<figure markdown>
  ![不同 shape 的 Tensor SDPA 算子的耗时](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/sdpa_performance.png)
  <figcaption>图 2：不同 shape 的 Tensor SDPA 算子的耗时（NVIDIA 4090）</figcaption>
</figure>

### 2.2 模型层面

当运行整个模型时，得益于 Client、Controller 和 Worker 异步计算的特性，DTorch 的 Python Client 执行时间比 PyTorch 少 64.17%，因此有充足的时间将系统调度的通信开销与计算重叠。同时，由于 CUDA Kernel 的 launch 更及时、更密集，还能小幅降低时延（当 Python 代码的运行耗时大于 CUDA Kernel 的执行耗时，GPU 会因等待而闲置，这种情况被称为"CPU 开销"）。

<figure markdown>
  |StableDiffusion3|PyTorch|DTorch|备注|
  |-|-|-|-|
  |Python Client 执行时间|0.575s|0.206s<span style="color: #4caf50; font-weight: bold;">(-64.17%)</span>|单卡推理的 CPU 耗时|
  |单卡耗时(GPU)|1.683s|1.648s<span style="color: #4caf50; font-weight: bold;">(-2.08%)</span>|单卡推理的端到端耗时|

  <figcaption>表 2：StableDiffusion3 单机单卡 1024*1024 4step 耗时（NVIDIA 3090）</figcaption>
</figure>

**Nsight System Profile**

下图是 Nsight System Profile 的分析结果。得益于 DTorch [`await TensorFuture`](../../user_guide/python_api_overview.md#5-异步获取-tensor-值) 的特性，CPU 开销得以与计算重叠，从而避免了：1. 获取模型输出 Tensor 的开销；2. 大量小算子导致的 CPU kernel launch 开销（text_encoder 的 input shape 很小，CPU 耗时远大于 GPU 耗时）。

> 关于 PyTorch 为何难以支持"异步获取 Tensor 的值"，详见 [Python API 使用指南 — PyTorch 中无法支持 await Tensor](../../user_guide/python_api_overview.md#pytorch-中无法支持-await-tensor)。

<figure markdown>
  ![Nsight System Profile 结果](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/nsys_sd3_async_get_tensor.png)
  <figcaption>图 3：StableDiffusion3 的 Nsight System Profile 结果</figcaption>
</figure>
