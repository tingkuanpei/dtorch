/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "tensor.h"

#include <torch/torch.h>

#include "dtorch/api/cpp/data_kind.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/functional/tensor_functional.h"
#include "dtorch/api/cpp/graph.h"
#include "dtorch/api/cpp/tensor_future.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operand.h"

namespace dtorch {
namespace api {
namespace cpp {

// api::cpp::Tensor rely on api::cpp::Graph, which meaning api::cpp::Graph must release after api::cpp::Tensor. This is
// easy to observe in c++, not in python. We have to use `nanobind::keep_alive<0, 1>()` to guarantee this.
// Call `nanobind::keep_alive<0, 1>()` is too tedious and fragile. So we wrap api::cpp::Graph
// in api::cpp::Tensor.
//
// # Python Example:
// class A:
//     def __init__(self) -> None:
//         print("Init A")
//     def __del__(self) -> None:
//         print("Del A")
// class B:
//     def __init__(self) -> None:
//         print("Init B")
//     def __del__(self) -> None:
//         print("Del B")
// a = A()
// b = B()
//
// # Execute result:
// Init A
// Init B
// Del A
// Del B

struct Tensor::Impl {
    Impl(Graph graph, const std::shared_ptr<core::Operand>& operand)
        : graph(graph),
          constructor(graph.GetGraphConstructor()),
          apiTensorRefCountHolder(constructor, operand.get()),
          operand(operand) {}

    DTORCH_API_DISABLE_COPY_AND_MOVE(Impl);

    Graph graph;
    core::GraphConstructor* constructor;
    core::ApiTensorRefCountHolder apiTensorRefCountHolder;
    std::shared_ptr<core::Operand> operand;
};

Tensor::Tensor(Graph& graph, const Shape& shape, const std::optional<DataKind>& dataKind,
               const std::optional<DeviceMesh>& deviceMesh, const std::optional<PlacementSeq>& placementSeq)
    : mImplPtr(functional::_Empty(graph, shape.Vec(), dataKind, deviceMesh, placementSeq).mImplPtr) {}

Tensor::Tensor(Graph& graph, torch::Tensor& torchTensor, const std::optional<DeviceMesh>& deviceMesh,
               const std::optional<PlacementSeq>& placementSeq)
    : mImplPtr(functional::_FromTorch(graph, torchTensor, deviceMesh, placementSeq).mImplPtr) {}

Tensor::~Tensor() = default;

void Tensor::SetName(const std::string& name) {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->SetName(name);
}

const std::string& Tensor::GetName() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->GetName();
}

const Shape& Tensor::GetShape() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->GetShape();
}

const Stride& Tensor::GetStride() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->GetStride();
}

DataKind Tensor::GetDataKind() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->GetDataKind();
}

bool Tensor::IsDistributed() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->IsDistributed();
}

DeviceKind Tensor::GetDeviceKind() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->GetDeviceMesh().GetDeviceKind();
}

const DeviceMesh& Tensor::GetDeviceMesh() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->GetDeviceMesh();
}

const PlacementSeq& Tensor::GetPlacementSeq() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand->GetPlacementSeq();
}

torch::Tensor Tensor::GetTorchTensor() const {
    DAlwaysAssert(mImplPtr != nullptr);
    // Delegate to async path, then block on Get()
    return GetTorchTensorAsync().Get();
}

void Tensor::InplaceAssignment(Tensor& other) { mImplPtr = other.mImplPtr; }

TensorFuture Tensor::GetTorchTensorAsync() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return functional::_GetTensorAsync(*this);
}

Tensor::Tensor(Graph graph, const std::shared_ptr<core::Operand>& operand)
    : mImplPtr(std::make_shared<Impl>(graph, operand)) {}

Graph& Tensor::GetGraph() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->graph;
}

std::shared_ptr<core::Operand>& Tensor::GetOperand() const {
    DAlwaysAssert(mImplPtr != nullptr);
    return mImplPtr->operand;
}

Tensor Tensor::GetNullTensorLike(const Tensor& tensor) {
    // Tensor with shape = {0} meaning null tensor
    Graph graph = tensor.mImplPtr->graph;
    // PlacementSeq for null tensor should be Placement::Optional(), so that PlacementSignature can match it.
    auto deviceMesh = tensor.GetDeviceMesh();
    PlacementSeq placements(deviceMesh.NumAxis(), Placement::Optional());
    return Tensor(graph, Shape::GetNullShape(), tensor.GetDataKind(), deviceMesh, placements);
}

void TensorArrayPushOptional(TensorArray& array) { TensorArrayPushOptional(array, std::nullopt); }

void TensorArrayPushOptional(TensorArray& array, const std::optional<Tensor>& tensor) {
    DAlwaysAssert(array.size() > 0);
    if (!tensor.has_value()) {
        array.push_back(Tensor::GetNullTensorLike(array[0]));
    } else {
        array.push_back(tensor.value());
    }
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
