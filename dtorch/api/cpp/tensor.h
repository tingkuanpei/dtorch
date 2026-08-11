/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#include "api_type.h"
#include "data_kind.h"
#include "device.h"
#include "distributed_spec.h"
#include "graph.h"
#include "memory"
#include "shape.h"
#include "stride.h"
#include "tensor_future.h"

namespace dtorch {
namespace core {
class Operand;
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace api {
namespace cpp {

class Tensor {
public:
    Tensor(Graph& graph, const Shape& shape, const std::optional<DataKind>& dataKind = std::nullopt,
           const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
           const std::optional<PlacementSeq>& placementSeq = std::nullopt);

    Tensor(Graph& graph, torch::Tensor& torchTensor, const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
           const std::optional<PlacementSeq>& placementSeq = std::nullopt);

    ~Tensor();

    void SetName(const std::string& name);

    const std::string& GetName() const;

    const Shape& GetShape() const;

    const Stride& GetStride() const;

    DataKind GetDataKind() const;

    bool IsDistributed() const;

    DeviceKind GetDeviceKind() const;

    const DeviceMesh& GetDeviceMesh() const;

    const PlacementSeq& GetPlacementSeq() const;

    torch::Tensor GetTorchTensor() const;

    TensorFuture GetTorchTensorAsync() const;

    Graph& GetGraph() const;

    // In-place assignment swaps the internal Impl pointer, making `this` share the same underlying
    // Operand/Graph state as `other`. This enables Python's `param.data = param.data.to(torch.float32)`
    // pattern where the Python wrapper object identity is preserved while the C++ internals are replaced.
    void InplaceAssignment(Tensor& other);

public:
    //------------------------------------------ Internal Function -----------------------------------------------------
    // Function in this part only call from internal

    Tensor(Graph graph, const std::shared_ptr<core::Operand>& operand);

    std::shared_ptr<core::Operand>& GetOperand() const;

    static Tensor GetNullTensorLike(const Tensor& tensor);

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;
};

using TensorArray = std::vector<Tensor>;

void TensorArrayPushOptional(TensorArray& array);

void TensorArrayPushOptional(TensorArray& array, const std::optional<Tensor>& tensor);

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
