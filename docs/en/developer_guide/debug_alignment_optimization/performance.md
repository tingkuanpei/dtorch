# Performance

## 1. Memory usage

Thanks to the asynchronous computation of the Client, Controller and Worker, DTorch's intermediate Tensors during computation can be released in time, so its memory usage is lower than PyTorch's. The measured data for the StableDiffusion3 model:

<figure markdown>
  ||PyTorch|DTorch|
  |-|-|-|
  |Peak usage|18.131GB|17.663GB<span style="color: #4caf50; font-weight: bold;">(-2.58%)</span>|

  <figcaption>Table 1: Peak memory of StableDiffusion3, single machine single GPU, 1024*1024, 4 steps</figcaption>
</figure>

The code snippet below explains the reason for this difference: the variables a, b, c are intermediate variables of the computation; because they are not released in time, they push up PyTorch's peak memory. DTorch uses asynchronous computation: the Python thread building the graph finishes `func()` and returns ahead of time, from which the C++ execution engine learns that the intermediate variables a, b, c are no longer held and can release their memory in time.

```python
def func():
    a = torch.rand(...)
    b = operator0(a)
    c = operator1(b)
    d = operator2(c)
    return d
```

## 2. Performance

### 2.1 Operator level

In theory, Single-Controller's scheduling overhead is higher than Multi-Controller's, but thanks to DTorch's deep optimization at the C++ layer, DTorch's overhead on small operators is only 37% higher than PyTorch's, and on large operators the two have essentially the same time.

The figure below shows the time of addition on Tensors of different shapes on CPU and GPU. Even for the addition of a Tensor with `Shape=(1,)`, DTorch is only 37% higher than PyTorch; as the Tensor grows, the times of the two become essentially equal. The same conclusion holds for the compute-intensive SDPA operator.

<figure markdown>
  ![Time of the add operator on Tensors of different shapes](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/tensor_add_performance.png)
  <figcaption>Figure 1: Time of the Tensor add operator at different shapes (NVIDIA 4090)</figcaption>
</figure>

<figure markdown>
  ![Time of the SDPA operator on Tensors of different shapes](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/sdpa_performance.png)
  <figcaption>Figure 2: Time of the Tensor SDPA operator at different shapes (NVIDIA 4090)</figcaption>
</figure>

### 2.2 Model level

When running a whole model, thanks to the asynchronous computation of the Client, Controller and Worker, DTorch's Python Client execution time is 64.17% less than PyTorch's, leaving ample time to overlap the communication overhead of system scheduling with computation. At the same time, because the CUDA Kernel launches are more timely and denser, the latency is also slightly reduced (when the Python code takes longer than the CUDA Kernels, the GPU idles waiting — this is called "CPU overhead").

<figure markdown>
  |StableDiffusion3|PyTorch|DTorch|Notes|
  |-|-|-|-|
  |Python Client execution time|0.575s|0.206s<span style="color: #4caf50; font-weight: bold;">(-64.17%)</span>|CPU time of single-GPU inference|
  |Single-GPU time (GPU)|1.683s|1.648s<span style="color: #4caf50; font-weight: bold;">(-2.08%)</span>|end-to-end time of single-GPU inference|

  <figcaption>Table 2: StableDiffusion3, single machine single GPU, 1024*1024, 4 steps (NVIDIA 3090)</figcaption>
</figure>

**Nsight System Profile**

The figure below is the Nsight System Profile analysis result. Thanks to DTorch's [`await TensorFuture`](../../user_guide/python_api_overview.md#5-async-tensor-value-retrieval) feature, the CPU overhead can be overlapped with computation, thereby avoiding: 1. the overhead of retrieving the model's output Tensor; 2. the CPU kernel launch overhead caused by a large number of small operators (the text_encoder's input shape is very small, so the CPU time far exceeds the GPU time).

> For why PyTorch finds it hard to support "retrieving a Tensor's value asynchronously", see [Python API Overview — Why PyTorch cannot support await on Tensor](../../user_guide/python_api_overview.md#why-pytorch-cannot-support-await-on-tensor).

<figure markdown>
  ![Nsight System Profile result](https://cdn.jsdelivr.net/gh/tingkuanpei/dtorch-asset@main/blog/nsys_sd3_async_get_tensor_en.png)
  <figcaption>Figure 3: Nsight System Profile result of StableDiffusion3</figcaption>
</figure>
