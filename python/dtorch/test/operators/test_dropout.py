"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_dropout(test_case, shape, device, dtype):
    torch_in = torch.rand(*shape, device=device, dtype=dtype)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_out = torch.nn.functional.dropout(torch_in, training=False, inplace=True)
    dtorch_out = dtorch.nn.functional.dropout(dtorch_in, training=False, inplace=True)
    assert_tensor_equal(test_case, torch_out, dtorch_out)

    torch_out = torch.nn.functional.dropout(torch_in, training=False, inplace=False)
    dtorch_out = dtorch.nn.functional.dropout(dtorch_in, training=False, inplace=False)
    assert_tensor_equal(test_case, torch_out, dtorch_out)


class TestDropout(unittest.TestCase):
    def test_dropout(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [torch.float32]
        for arg in gen_arg_list(arg_dict):
            _test_dropout(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
