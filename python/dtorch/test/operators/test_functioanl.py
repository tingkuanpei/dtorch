"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest

import torch
import torch.nn.functional as TorchF

import dtorch
import dtorch
import dtorch.nn.functional as DTorchF
from dtorch.test.test_util import (
    check_pytorch_version,
)


class TestFunctional(unittest.TestCase):
    def test_functional_interface(test_case):
        skip_func_name_set = ()
        if not check_pytorch_version(min_version="2.5.0"):
            skip_func_name_set = skip_func_name_set + ("rms_norm",)

        for original_function_name in DTorchF._function_names:
            if original_function_name in skip_func_name_set:
                continue
            function_name = original_function_name

            if function_name.startswith("_"):
                function_name = function_name.lstrip("_")
                if hasattr(TorchF, function_name) and not hasattr(DTorchF, function_name):
                    error_msg = f"{function_name} in torch.nn.functional, but it is private"
                    test_case.assertFalse(True, msg=error_msg)
            else:
                error_msg = f"{function_name} not in torch.nn.functional, it shoud begin with '_'"
                test_case.assertTrue(hasattr(TorchF, function_name), msg=error_msg)

            if hasattr(torch, function_name):
                error_msg = f"{function_name} shoud in ugrpah"
                test_case.assertTrue(hasattr(dtorch, function_name), msg=error_msg)

            if hasattr(torch.Tensor, function_name):
                error_msg = f"{function_name} shoud in ugrpah.Tensor"
                test_case.assertTrue(hasattr(dtorch.Tensor, function_name), msg=error_msg)


if __name__ == "__main__":
    unittest.main()
