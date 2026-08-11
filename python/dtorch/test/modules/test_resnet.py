"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest
from collections import OrderedDict


import torch
import torchvision
import time

import dtorch
from dtorch.test.test_util import assert_tensor_allclose, gen_arg_list
from dtorch.test.modules.resnet import resnet50


@torch.inference_mode
def _test_resnet(test_case, device):
    torch_in = torch.randn(1, 3, 64, 64)
    torch_in = torch_in.to(device=device)
    torch_resnet = torchvision.models.resnet50().eval()
    torch_resnet.to(device)
    torch_out = torch_resnet(torch_in)
    torch_out = torch_out.detach()

    dtorch_in = dtorch.Tensor(torch_in)
    dtorch_resnet = resnet50()
    dtorch_resnet.load_state_dict(torch_resnet.state_dict())
    dtorch_resnet.to(device=device)
    dtorch_out = dtorch_resnet(dtorch_in)

    assert_tensor_allclose(test_case, torch_in, dtorch_in)
    assert_tensor_allclose(test_case, torch_out, dtorch_out, rtol=1e-4, atol=1e-4)


class TestResNet(unittest.TestCase):
    def test_resnet(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in gen_arg_list(arg_dict):
            _test_resnet(test_case, *arg)

    def tearDown(self):
        dtorch.default_graph.empty_cache()

    # def test_resnet_speed(test_case):
    #     warmup_count = 3
    #     loop_count = 10

    #     # torch
    #     torch_in = torch.randn(1, 3, 224, 224).cuda()
    #     torch_resnet = torchvision.models.resnet50().to("cuda").eval()
    #     for _ in range(warmup_count):
    #         torch_out = torch_resnet(torch_in)

    #     torch.cuda.synchronize()
    #     torch.cuda.nvtx.range_push("PyTorch resnet")
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     for i in range(loop_count):
    #         dtorch.cuda.nvtx.range_push("PyTorch resnet: {}".format(i))
    #         torch_out = torch_resnet(torch_in)
    #         dtorch.cuda.nvtx.range_pop()
    #     end.record()
    #     torch.cuda.synchronize()
    #     torch.cuda.nvtx.range_pop()
    #     torch_duration = start.elapsed_time(end) / loop_count
    #     print("torch resnet50 time: {} ms".format(torch_duration))
    #     torch_out = torch_out.detach().cpu()

    #     # dtorch
    #     dtorch_in = from_torch_tensor(torch_in)
    #     dtorch_resnet = resnet50()
    #     dtorch_resnet.load_state_dict(torch_resnet.state_dict())
    #     dtorch_resnet.to(torch.device("gpu"))
    #     for _ in range(warmup_count):
    #         dtorch_out = dtorch_resnet(dtorch_in)

    #     dtorch.Graph.default_graph().sync()
    #     dtorch.cuda.nvtx.range_push("DTorch resnet")
    #     start = time.time()
    #     for i in range(loop_count):
    #         dtorch.cuda.nvtx.range_push("DTorch resnet: {}".format(i))
    #         dtorch_out = dtorch_resnet(dtorch_in)
    #         dtorch.cuda.nvtx.range_pop()
    #     dtorch.Graph.default_graph().sync()
    #     dtorch.cuda.nvtx.range_pop()
    #     end = time.time()
    #     dtorch_duration = (end - start) / loop_count * 1000
    #     print("dtorch resnet50 time: {} ms".format(dtorch_duration))

    #     # check result equal
    #     assert_tensor_allclose(test_case, torch_out, dtorch_out)


if __name__ == "__main__":
    unittest.main()
