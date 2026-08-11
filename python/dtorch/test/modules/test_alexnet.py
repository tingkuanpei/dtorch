"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest

import torch
import torchvision

import dtorch
from dtorch.test.test_util import assert_tensor_allclose
from dtorch.test.modules.alexnet import AlexNet


class TestAlexNet(unittest.TestCase):
    @torch.inference_mode
    def test_alexnet(test_case):
        torch_in = torch.randn(1, 3, 224, 224)
        torch_alexnet = torchvision.models.alexnet().eval()
        torch_out = torch_alexnet(torch_in)
        torch_out = torch_out.detach()

        dtorch_in = dtorch.Tensor(torch_in)
        dtorch_alexnet = AlexNet()
        dtorch_alexnet.load_state_dict(torch_alexnet.state_dict())
        dtorch_out = dtorch_alexnet(dtorch_in)

        assert_tensor_allclose(test_case, torch_in, dtorch_in)
        assert_tensor_allclose(test_case, torch_alexnet.features[0].weight, dtorch_alexnet.features[0].weight)
        assert_tensor_allclose(test_case, torch_out, dtorch_out)

    def tearDown(self):
        dtorch.default_graph.empty_cache()


if __name__ == "__main__":
    unittest.main()
