"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import os
from typing import Union, List, Sequence
from enum import Enum
import functools

import torch
import numpy as np

import dtorch


# TensorChecker is used to verify that DTorch distributed modules produce the same output as
# their PyTorch reference modules. It works by registering tensors under the same tag from both
# sides, then calling check_equal (exact) or check_allclose (approximate) to compare them.
#
# There are two ways to register tensors:
#
# 1. Manual registration via register_tensor:
#       from dtorch.util.tensor_checker import tensor_checker
#
#       # In PyTorch code:
#       tensor_checker.register_tensor("my_tensor", torch_tensor)
#
#       # In DTorch code:
#       tensor_checker.register_tensor("my_tensor", dtorch_tensor)
#
#       tensor_checker.check_equal()
#
# 2. Automatic registration via module_register_tensor_checker:
#       from dtorch.util.tensor_checker import tensor_checker, module_register_tensor_checker
#
#       module_register_tensor_checker(torch_model, tensor_checker)
#       module_register_tensor_checker(dtorch_model, tensor_checker)
#       # run model forward — parameters, buffers, inputs, and outputs are auto-registered
#       tensor_checker.check_equal()
#       tensor_checker.check_allclose()
#
#    NOTE: module_register_tensor_checker only compares module-level parameters, buffers,
#    inputs, and outputs. It requires that the corresponding submodules in the two models
#    have identical names (via named_modules), otherwise they cannot be matched and compared.
#
# DumpMode controls whether mismatched tensors are saved to disk for debugging:
#     DumpMode.NOT              - never dump (default)
#     DumpMode.ALL              - dump all tensors
#     DumpMode.NOT_EQUAL        - dump only tensors that differ
#     DumpMode.FIRST_NOT_EQUAL  - dump only the first group of tensors that differ
#
#     tensor_checker.check_equal(dump_mode=DumpMode.NOT_EQUAL, dump_path="/tmp/debug")


class DumpMode(Enum):
    NOT = 1
    ALL = 2
    NOT_EQUAL = 3
    FIRST_NOT_EQUAL = 4


class TensorChecker:
    def __init__(self):
        self.tensor_map = {}

    def clear(self):
        self.tensor_map.clear()

    def register_tensor(self, tag: str, tensor: Union[torch.Tensor, dtorch.Tensor, np.ndarray]) -> None:
        assert (
            isinstance(tensor, torch.Tensor) or isinstance(tensor, dtorch.Tensor) or isinstance(tensor, np.ndarray)
        ), f"Get error type: {type(tensor)}"
        if isinstance(tensor, dtorch.Tensor):
            tensor = tensor.to_torch()
        elif isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor.copy())

        if tag not in self.tensor_map:
            self.tensor_map[tag] = []
        self.tensor_map[tag].append(tensor.cpu().clone())

    def tensor_from_registered(self, tag: str, tensor: Union[torch.Tensor, dtorch.Tensor]) -> "Tensor":
        assert isinstance(tensor, torch.Tensor) or isinstance(tensor, dtorch.Tensor), f"Get error type: {type(tensor)}"
        assert tag in self.tensor_map
        assert len(self.tensor_map[tag]) > 0

        src_tensor = self.tensor_map[tag][0]
        assert src_tensor.dtype == tensor.dtype
        if isinstance(tensor, torch.Tensor):
            return src_tensor.to(device=tensor.device)
        else:
            return dtorch.Tensor(src_tensor).redistribute_like(tensor)

    def _check_impl(self, compare_fn, dump_mode: DumpMode, dump_path: str) -> bool:
        if dump_mode != DumpMode.NOT and not os.path.exists(dump_path):
            os.makedirs(dump_path)

        result = True
        not_equal_count = 0
        for key, tensors in self.tensor_map.items():
            assert isinstance(tensors, list)
            if len(tensors) < 2:
                print(f"Number of tensors in self.tensor_map[{key}] less than 2")
                continue

            tensor = tensors[0]
            equal_in_key = True
            for other_tensor in tensors[1:]:
                if not compare_fn(tensor, other_tensor):
                    print(f"TensorChecker: {len(tensors)} tensors not equal in {key}")
                    result = False
                    equal_in_key = False
                    not_equal_count = not_equal_count + 1
            if equal_in_key:
                print(f"TensorChecker: {len(tensors)} tensors equal in {key}")
            if (
                dump_mode == DumpMode.ALL
                or (dump_mode == DumpMode.NOT_EQUAL and not equal_in_key)
                or (dump_mode == DumpMode.FIRST_NOT_EQUAL and not_equal_count == 1)
            ):
                self.dump_tensors(dump_path, key, tensors)
        return result

    def check_equal(
        self,
        dump_mode: DumpMode = DumpMode.NOT,
        dump_path: str = "",
    ) -> bool:
        return self._check_impl(TensorChecker.tensor_equal, dump_mode, dump_path)

    def check_allclose(
        self,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        dump_mode: DumpMode = DumpMode.NOT,
        dump_path: str = "",
    ) -> bool:
        compare_fn = lambda a, b: TensorChecker.tensor_allclose(a, b, rtol, atol, equal_nan)
        return self._check_impl(compare_fn, dump_mode, dump_path)

    def dump_tensors(self, dump_path, key, tensors):
        for i, tensor in enumerate(tensors):
            path = os.path.join(dump_path, f"{key}_idx_{i}.npy")
            numpy_tensor = tensor.cpu().numpy()
            np.save(path, numpy_tensor)

    @staticmethod
    def better_print(tensor, other):
        tensor_str = str(tensor).split("\n")
        other_str = str(other).split("\n")
        max_length = max([len(a) for a in tensor_str]) + 10
        empty_pefix = " " * (max_length // 2 - 10)

        print_str = []
        print_str.append("-" * 100)
        print_str.append((empty_pefix + "input a").ljust(max_length) + "|   " + (empty_pefix + "input b"))
        for a, b in zip(tensor_str, other_str):
            print_str.append(a.ljust(max_length) + "|   " + b)
        print_str.append("-" * 100)
        print_str = "\n".join(print_str)
        print(print_str)

    @staticmethod
    def convert_and_base_check(
        a: Union[torch.Tensor, dtorch.Tensor, np.ndarray],
        b: Union[torch.Tensor, dtorch.Tensor, np.ndarray],
        print_no_equal_msg: bool = True,
    ) -> bool:
        def to_torch(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor
            elif isinstance(tensor, dtorch.Tensor):
                return tensor.to_torch()
            elif isinstance(tensor, np.ndarray):
                return torch.from_numpy(tensor.copy())
            else:
                raise TypeError(f"unsupported tensor type: {type(tensor)}")

        a = to_torch(a)
        b = to_torch(b)

        if not a.dtype == b.dtype:
            if print_no_equal_msg:
                print(f"Tensor dtype not equal: {a.dtype} vs {b.dtype}")
            return False, a, b

        if not a.device == b.device:
            if print_no_equal_msg:
                print(f"Tensor device not equal: {a.device} vs {b.device}")
            return False, a, b

        if not a.shape == b.shape:
            if print_no_equal_msg:
                print(f"Tensor shape not equal: {a.shape} vs {b.shape}")
            return False, a, b

        if a.dtype in {torch.uint8, torch.uint16, torch.uint32, torch.uint64}:
            a = a.float()
            b = b.float()

        return True, a, b

    @staticmethod
    def tensor_equal(
        a: Union[torch.Tensor, dtorch.Tensor, np.ndarray],
        b: Union[torch.Tensor, dtorch.Tensor, np.ndarray],
        print_no_equal_msg: bool = True,
    ) -> bool:
        same_flag, a, b = TensorChecker.convert_and_base_check(a, b, print_no_equal_msg)
        if not same_flag:
            return same_flag

        equal = torch.equal(a, b)
        if not equal and print_no_equal_msg:
            print("======================================== Tensor not equal =========================================")
            TensorChecker.better_print(a, b)
            diff_mask = a != b
            max_idx = torch.argmax(
                torch.where(diff_mask, torch.ones_like(a, dtype=torch.int64), torch.zeros_like(a, dtype=torch.int64))
            )
            a_flat = a.flatten()
            b_flat = b.flatten()
            print(
                f"    first different index: {max_idx}, value: {a_flat[max_idx]}  {b_flat[max_idx]}, dtype: ({a.dtype})"
            )
            print(f"    nearby index:")
            nearby_range = 5
            for i in range(nearby_range * 2):
                idx = max_idx - nearby_range + i
                if idx >= 0 and idx < a_flat.numel():
                    mark = " <--" if idx == max_idx else ""
                    print(f"                  index: {idx}, value: {a_flat[idx]}  {b_flat[idx]}{mark}")
            print(
                "===================================================================================================="
            )
        return equal

    @staticmethod
    def tensor_meanclose(
        a: Union[torch.Tensor, dtorch.Tensor, np.ndarray],
        b: Union[torch.Tensor, dtorch.Tensor, np.ndarray],
        rtol: float = 1e-05,
        atol: float = 1e-08,
        print_no_equal_msg: bool = True,
    ) -> bool:
        same_flag, a, b = TensorChecker.convert_and_base_check(a, b, print_no_equal_msg)
        if not same_flag:
            return same_flag

        mean_error = (a - b).abs().mean()
        equal = mean_error <= (b.mean() * rtol + atol)
        if not equal and print_no_equal_msg:
            print("======================================== Tensor not equal =========================================")
            TensorChecker.better_print(a, b)
            print(f"mean_error: {mean_error}, a.mean: {a.mean()}, b.mean: {b.mean()},")
            print(
                "===================================================================================================="
            )
        return equal

    @staticmethod
    def tensor_allclose(
        a: Union[torch.Tensor, dtorch.Tensor, np.ndarray],
        b: Union[torch.Tensor, dtorch.Tensor, np.ndarray],
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        print_no_equal_msg: bool = True,
    ) -> bool:
        same_flag, a, b = TensorChecker.convert_and_base_check(a, b, print_no_equal_msg)
        if not same_flag:
            return same_flag

        equal = torch.allclose(a, b, rtol, atol, equal_nan)
        if not equal and print_no_equal_msg:
            print("======================================== Tensor not equal =========================================")
            TensorChecker.better_print(a, b)
            if equal_nan:
                both_nan_mask = torch.isnan(a) & torch.isnan(b)
                a[both_nan_mask] = 0
                b[both_nan_mask] = 0
            max_idx = torch.argmax(torch.abs(a - b))
            idx = max_idx
            a = a.flatten()
            b = b.flatten()
            print(f"    max different index: {idx}, value: {a[idx]}  {b[idx]}, dtype: ({a.dtype})")
            print(f"    nearby index:")
            nearby_range = 5
            for i in range(nearby_range * 2):
                idx = max_idx - nearby_range + i
                if idx >= 0 and idx < a.numel():
                    print(f"                  index: {idx}, value: {a[idx]}  {b[idx]}")
            print(
                "===================================================================================================="
            )
        return equal


tensor_checker = TensorChecker()


def module_forward_decorator(func, module, checker, base_tag):
    # Module may repeat mutil time
    module.__dtorch_tensor_checker_visit_count__ = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if module.__dtorch_tensor_checker_visit_count__ == 0:
            visit_count_string = ""
        else:
            visit_count_string = f"_revisit_{module.__dtorch_tensor_checker_visit_count__}"

        # Register module parameter
        named_parameters = dict(module.named_parameters(recurse=False))
        for name, param in named_parameters.items():
            checker.register_tensor(f"{base_tag}{visit_count_string}_{name}", param)

        # Register module buffer
        named_buffers = dict(module.named_buffers(recurse=False))
        for name, buffer in named_buffers.items():
            checker.register_tensor(f"{base_tag}{visit_count_string}_{name}", buffer)

        # Register module input
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) or isinstance(arg, dtorch.Tensor):
                checker.register_tensor(f"{base_tag}{visit_count_string}_input_{idx}", arg)

        for name, arg in kwargs.items():
            if isinstance(arg, torch.Tensor) or isinstance(arg, dtorch.Tensor):
                checker.register_tensor(f"{base_tag}{visit_count_string}_input_{name}", arg)

        outputs = func(*args, **kwargs)
        if not isinstance(outputs, list) and not isinstance(outputs, tuple):
            outputs_list = [
                outputs,
            ]
        else:
            outputs_list = outputs

        # Register module output
        for idx, output in enumerate(outputs_list):
            if isinstance(output, torch.Tensor) or isinstance(output, dtorch.Tensor):
                checker.register_tensor(f"{base_tag}{visit_count_string}_output_{idx}", output)

        module.__dtorch_tensor_checker_visit_count__ += 1
        return outputs

    return wrapper


def module_register_tensor_checker(
    modules: List[Union[torch.nn.Module, dtorch.nn.Module]],
    checker: TensorChecker = None,
):
    if checker is None:
        checker = tensor_checker

    if not isinstance(modules, Sequence):
        modules = [modules]

    named_modules_list = [dict(m.named_modules(remove_duplicate=True)) for m in modules]
    names_list = [name_and_module.keys() for name_and_module in named_modules_list]
    common_names = set(set.intersection(*(set(names) for names in names_list)))

    for named_modules in named_modules_list:
        for name, module in named_modules.items():
            if name not in common_names:
                continue

            module.forward = module_forward_decorator(module.forward, module, checker, f"module_checker#{name}")
