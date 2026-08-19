# 精度对齐（Precision Alignment）

DTorch 中调用 LibTorch，而 PyTorch 则通过 python 调用 LibTorch，两者运行的 Kernel 一致，输出应当时一致的。但也有一些情况，使得两者的输出不一致。

## View + Linear
在 NIVIDA 3090 上遇到 View + Linear 可能会导致输出不一致的情况。代码如下，当给 PyTorch 和 DTorch 都添加 `hidden_states = hidden_states.contiguous()` 后，两种调用的 Kernel 一致。

```python
# https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/models/attention_processor.py#L1488
#
# 在 slice 之后，hidden_states 的以下属性均一致：
#   hidden_states.stride()
#   hidden_states.untyped_storage().nbytes()
#   hidden_states.numel() * hidden_states.element_size()
# 仅发现 _is_view() 属性不一致。
#   PyTorch: hidden_states._is_view() = False
#   DTorch: hidden_states._is_view() = True
#
# 之后在 Linear 中，PyTorch 调用了三个 Kernel，DTorch 则调用两个 Kernel。
#   PyTorch:
#       void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
#       ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn
#       void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
#
#   DTorch
#       ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn
#       void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)

hidden_states, encoder_hidden_states = (
    hidden_states[:, : residual.shape[1]],
    hidden_states[:, residual.shape[1] :],
)

# attn.to_out[0] = torch.nn.Linear
hidden_states = attn.to_out[0](hidden_states)
```
