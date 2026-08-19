# 性能优化

## 1. CUDA Kernel Launch 速度慢

### 问题描述

DTorch 使用独立的 kernel stream 线程来调用 LibTorch 接口并 launch CUDA kernel。CUDA stream 内部维护了一个 launch queue，有容量上限。当 launch 速率超过队列容量时，CPU 线程会被阻塞，直至队列有空位为止。此时表现出来的现象就是 launch CUDA kernel 的速度显著变慢。

### 诊断方法

通过 `CUDA_SCALE_LAUNCH_QUEUES` 环境变量可以调整 CUDA stream queue 的容量大小。如果增大该值后 kernel launch 耗时明显下降，则说明当前瓶颈确实是 stream queue 容量不足导致的阻塞。

```bash
export CUDA_SCALE_LAUNCH_QUEUES=4x
```

### 注意事项

`CUDA_SCALE_LAUNCH_QUEUES` 主要用作诊断手段，通常修改此环境变量不会带来实质性的性能提升。该变量只是调整队列容量上限，并不改变实际的 launch 吞吐能力。如果确实命中此瓶颈，更根本的解决思路是优化 launch 模式本身，例如减少 launch 次数、使用 CUDA Graph 合并 kernel 等。

### 参考链接

- [CUDA Environment Variables — CUDA_SCALE_LAUNCH_QUEUES](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-scale-launch-queues)
