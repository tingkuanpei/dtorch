"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict

import torch

import dtorch
from dtorch.test.test_util import gen_arg_list, assert_tensor_equal


def _test_chunk(test_case, device):
    shape = (6, 6, 8)
    torch_in = torch.rand(*shape, dtype=torch.float32, device=device)
    dtorch_in = dtorch.Tensor(torch_in)

    torch_outs = torch_in.chunk(4, 0)
    dtorch_outs = dtorch_in.chunk(4, 0)
    assert_tensor_equal(test_case, torch_outs, dtorch_outs)

    torch_outs = torch_in.unbind(1)
    dtorch_outs = dtorch_in.unbind(1)
    assert_tensor_equal(test_case, torch_outs, dtorch_outs)


class TestChunk(unittest.TestCase):
    def test_chunk(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_chunk(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
