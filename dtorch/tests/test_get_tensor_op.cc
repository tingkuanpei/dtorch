/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include <torch/torch.h>

#include "dtorch/api/cpp/functional/tensor_functional.h"
#include "dtorch/api/cpp/graph.h"
#include "dtorch/api/cpp/tensor.h"
#include "dtorch/api/cpp/tensor_future.h"
#include "dtorch/core/operators/system/get_tensor_op.h"
#include "test.h"

using dtorch::api::cpp::Graph;
using dtorch::api::cpp::Tensor;
using dtorch::api::cpp::TensorFuture;
using dtorch::api::cpp::functional::_GetTensorAsync;

// ============================================================
// GetTensorOp tests
// ============================================================

TEST(GetTensorOpTest, BasicCompute) {
    Graph graph;
    auto torchTensor = torch::ones({3, 4});
    Tensor dtorchTensor(graph, torchTensor);

    // Sync get (exercise the full path)
    auto result = dtorchTensor.GetTorchTensor();
    ASSERT_TRUE(torch::equal(torchTensor, result));
}

TEST(GetTensorOpTest, AsyncGetBasic) {
    Graph graph;
    auto torchTensor = torch::rand({2, 5});
    Tensor dtorchTensor(graph, torchTensor);

    // Async get
    TensorFuture future = dtorchTensor.GetTorchTensorAsync();

    // Block on result
    auto result = future.Get();
    ASSERT_TRUE(torch::equal(torchTensor, result));
}

TEST(GetTensorOpTest, AsyncGetWait) {
    Graph graph;
    auto torchTensor = torch::zeros({4, 4});
    Tensor dtorchTensor(graph, torchTensor);

    TensorFuture future = dtorchTensor.GetTorchTensorAsync();

    // Wait with sufficient timeout
    ASSERT_TRUE(future.WaitFor(5000));
    auto result = future.Get();
    ASSERT_TRUE(torch::equal(torchTensor, result));
}

TEST(GetTensorOpTest, SyncAndAsyncConsistency) {
    Graph graph;
    auto torchTensor = torch::rand({3, 3});
    Tensor dtorchTensor(graph, torchTensor);

    auto syncResult = dtorchTensor.GetTorchTensor();
    auto asyncResult = dtorchTensor.GetTorchTensorAsync().Get();

    ASSERT_TRUE(torch::equal(syncResult, asyncResult));
    ASSERT_TRUE(torch::equal(torchTensor, syncResult));
}

TEST(GetTensorOpTest, GetTensorAsyncFromFunctional) {
    Graph graph;
    auto torchTensor = torch::ones({2, 2});
    Tensor dtorchTensor(graph, torchTensor);

    TensorFuture future = _GetTensorAsync(dtorchTensor);
    auto result = future.Get();
    ASSERT_TRUE(torch::equal(torchTensor, result));
}
