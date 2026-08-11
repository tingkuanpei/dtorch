/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <torch/torch.h>

#include "test.h"

TEST(TorchKernelTest, SimpleTest) {
    torch::Tensor tensor = torch::rand({4, 3});
    auto viewedTensor = tensor.view({12, 1});
    auto indexTensor = tensor[0];
    auto slicedTensor = tensor.slice(0, 1, 3);

    tensor[0][0] = 1.0f;

    EXPECT_TRUE(tensor.index({0, 0}).item<float>() == 1.0f);
    EXPECT_TRUE(indexTensor.index({0}).item<float>() == 1.0f);

    torch::ScalarType scalarType = torch::kFloat32;

    auto options = torch::TensorOptions().dtype(scalarType).device(torch::kCPU).requires_grad(true);

    torch::Tensor tensor1 = torch::empty({3, 4}, options);
    EXPECT_TRUE(tensor1.dtype() == torch::kFloat32);
    EXPECT_TRUE(tensor1.device().type() == torch::kCPU);
    EXPECT_TRUE(tensor1.requires_grad());
}

class TestMyFunction : public torch::autograd::Function<TestMyFunction> {
public:
    static constexpr bool is_traceable = true;

    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, const torch::Tensor& input, int n) {
        ctx->saved_data["n"] = n;
        return input.mul(n);
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::tensor_list grad_output) {
        auto n = ctx->saved_data["n"].toInt();
        return {grad_output[0] * n, torch::Tensor()};
    }
};

TEST(TorchKernelTest, AutoGradientTest) {
    torch::Tensor x = torch::rand({4, 3}).requires_grad_(true);
    torch::Tensor y = TestMyFunction::apply(x, 6);
    y.sum().backward();

    torch::Tensor expectGradient = torch::full({4, 3}, 6.0);
    EXPECT_TRUE(torch::allclose(expectGradient, x.grad()));
}
