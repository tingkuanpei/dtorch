"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import inspect
import time
import psutil
import asyncio

import torch

import dtorch


class Metric:
    def __init__(self, duration, torch_gpu_memory, dtorch_gpu_memory, cpu_memory):
        self.duration = duration
        self.torch_gpu_memory = torch_gpu_memory
        self.dtorch_gpu_memory = dtorch_gpu_memory
        self.cpu_memory = cpu_memory

    def __repr__(self) -> str:
        return f"duration: {self.duration}, torch_gpu_memory: {self.torch_gpu_memory}, dtorch_gpu_memory: {self.dtorch_gpu_memory}, cpu_memory: {self.cpu_memory}"


class Benchmark:
    @staticmethod
    def run(func, warmup=1, count=3):
        dtorch.default_graph.empty_cache()

        sig = inspect.signature(func)
        accepts_benchmark_last_round = "benchmark_last_round" in sig.parameters

        if inspect.iscoroutinefunction(func):
            # Use a single event loop for all iterations so that asyncio state
            # (e.g. Tasks stored via nonlocal) can be carried across calls.
            async def _run_async():
                for i in range(warmup):
                    if accepts_benchmark_last_round and i == warmup - 1:
                        await func(benchmark_last_round=True)
                    else:
                        await func()

                dtorch.default_graph.sync()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin_time = time.time()

                func_out = None
                for i in range(count):
                    if accepts_benchmark_last_round and i == count - 1:
                        func_out = await func(benchmark_last_round=True)
                    else:
                        func_out = await func()

                dtorch.default_graph.sync()
                torch.cuda.synchronize()
                end_time = time.time()
                duration_time = (end_time - begin_time) / count

                torch_gpu_memory = torch.cuda.max_memory_allocated() / 1e6
                dtorch_gpu_memory = (
                    dtorch.default_graph.get_memory_stats(reset_peak=True).find(0).max_memory_allocated / 1e6
                )
                cpu_memory = psutil.Process().memory_info().rss / 1e6

                dtorch.default_graph.empty_cache()
                return Metric(duration_time, torch_gpu_memory, dtorch_gpu_memory, cpu_memory), func_out

            return asyncio.run(_run_async())
        else:
            # Sync path
            for i in range(warmup):
                if accepts_benchmark_last_round and i == warmup - 1:
                    func(benchmark_last_round=True)
                else:
                    func()

            dtorch.default_graph.sync()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin_time = time.time()

            for i in range(count):
                if accepts_benchmark_last_round and i == count - 1:
                    func_out = func(benchmark_last_round=True)
                else:
                    func_out = func()

            dtorch.default_graph.sync()
            torch.cuda.synchronize()
            end_time = time.time()
            duration_time = (end_time - begin_time) / count

            torch_gpu_memory = torch.cuda.max_memory_allocated() / 1e6
            dtorch_gpu_memory = (
                dtorch.default_graph.get_memory_stats(reset_peak=True).find(0).max_memory_allocated / 1e6
            )
            cpu_memory = psutil.Process().memory_info().rss / 1e6

            dtorch.default_graph.empty_cache()
            return Metric(duration_time, torch_gpu_memory, dtorch_gpu_memory, cpu_memory), func_out

    @staticmethod
    def print(titles, metrics):
        assert len(titles) == len(metrics)

        title_length = max(len(t) for t in titles) + 2
        title_length = max(title_length, 20)
        total_length = title_length + 80

        print("=" * total_length)
        print(f"{'Benchmark Result':^{total_length}}")
        print("=" * total_length)
        print(
            f"{'titles':^{title_length}}|{'duration/s':^20}|{'torch_gpu_memory/MB':^20}|{'dtorch_gpu_memory/MB':^20}|{'cpu_memory/MB':^20}"
        )
        for title, metric in zip(titles, metrics):
            print(
                f"{title:^{title_length}}|{metric.duration:^20.4e}|{metric.torch_gpu_memory:^20.1f}|{metric.dtorch_gpu_memory:^20.1f}|"
                f"{metric.cpu_memory:^20.1f}"
            )
        print("=" * total_length)
