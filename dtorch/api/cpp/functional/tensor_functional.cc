/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "tensor_functional.h"

#include <csignal>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "dtorch/common/config.h"
#if DTORCH_WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

#include "dtorch/api/cpp/data_kind.h"
#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/functional/implement/convert_op_impl.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/communication/promise_future/tensor_promise_future.h"
#include "dtorch/core/graph/graph_constructor.h"
#include "dtorch/core/operators/standard/chunk_op.h"
#include "dtorch/core/operators/standard/clamp_op.h"
#include "dtorch/core/operators/standard/clone_op.h"
#include "dtorch/core/operators/standard/concat_op.h"
#include "dtorch/core/operators/standard/contiguous_op.h"
#include "dtorch/core/operators/standard/copy_op.h"
#include "dtorch/core/operators/standard/create_op.h"
#include "dtorch/core/operators/standard/expand_op.h"
#include "dtorch/core/operators/standard/get_item_op.h"
#include "dtorch/core/operators/standard/masked_op.h"
#include "dtorch/core/operators/standard/max_min_op.h"
#include "dtorch/core/operators/standard/pad_op.h"
#include "dtorch/core/operators/standard/permute_op.h"
#include "dtorch/core/operators/standard/reduce_op.h"
#include "dtorch/core/operators/standard/repeat_interleave_op.h"
#include "dtorch/core/operators/standard/repeat_op.h"
#include "dtorch/core/operators/standard/set_item_op.h"
#include "dtorch/core/operators/standard/squeeze_op.h"
#include "dtorch/core/operators/standard/transpose_op.h"
#include "dtorch/core/operators/standard/unsqueeze_op.h"
#include "dtorch/core/operators/standard/view_op.h"
#include "dtorch/core/operators/standard/where_op.h"
#include "dtorch/core/operators/system/get_tensor_op.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor CreateOpImpl(core::CreateKind createKind, Graph graph, const IntOrIntArray& shape,
                    const std::optional<DataKind>& dtype, const std::optional<DeviceMesh>& deviceMesh,
                    const std::optional<PlacementSeq>& placements,
                    const std::optional<Generator>& generator = std::nullopt, double doubleArg0 = 0.0f,
                    double doubleArg1 = 0.0f, double doubleArg2 = 0.0f,
                    const std::optional<torch::Tensor>& torchValue = std::nullopt) {
    // 1. Get resultDeviceMesh & resultPlacementSeq from arguments or graph default value
    std::optional<Graph> optGraph(graph);
    DeviceMesh resultDeviceMesh = deviceMesh.value_or(graph.GetDefaultDeviceMesh());
    PlacementSeq resultPlacementSeq;
    if (resultDeviceMesh.IsDistributed()) {
        resultPlacementSeq = placements.value_or(PlacementSeq(resultDeviceMesh.NumAxis(), Replicate()));
    } else {
        resultPlacementSeq = PlacementSeq(resultDeviceMesh.NumAxis(), Replicate());
    }

    // Validate placements count matches device mesh ndim (user-input check, before any internal assertions)
    if (placements.has_value() && placements->Size() != resultDeviceMesh.NumAxis()) {
        std::stringstream ss;
        ss << "Number of placements (" << placements->Size() << ") must match DeviceMesh ndim ("
           << resultDeviceMesh.NumAxis() << "). "
           << "Each axis of the DeviceMesh requires exactly one Placement (Shard/Replicate/Partial).";
        throw std::invalid_argument(ss.str());
    }
    resultPlacementSeq.ToReplicateWhenDimSizeEqualOne(resultDeviceMesh);

    // 2. Check if need redistribute
    // Some operator's values are different on different ranks, so we need to generate on rank 0 and redistribute to
    // other ranks.
    std::unordered_set<core::CreateKind> needRedistributeKind = {
        core::CreateKind::kArange, core::CreateKind::kRand, core::CreateKind::kRandInt,
        core::CreateKind::kRandn,  core::CreateKind::kEye,  core::CreateKind::kFromTorch,
    };
    bool needRedistribute = resultDeviceMesh.IsDistributed() && needRedistributeKind.count(createKind) > 0;
    DeviceMesh localDeviceMesh(Device(resultDeviceMesh.GetDeviceKind(), 0));
    PlacementSeq localPlacements(1, Replicate());

    // 3. Create CreateParam & add operator to graph
    std::unique_ptr<core::OpParam> param = nullptr;
    param = std::make_unique<core::CreateParam>(createKind, shape, dtype.value_or(graph.GetDefaultDataKind()),
                                                needRedistribute ? localDeviceMesh : resultDeviceMesh,
                                                needRedistribute ? localPlacements : resultPlacementSeq, generator,
                                                doubleArg0, doubleArg1, doubleArg2, torchValue);
    Tensor result = core::GraphConstructor::AddOperator(std::move(param), api::cpp::TensorArray(), optGraph);

    // 4. Redistribute if needed
    if (needRedistribute) {
        result = _Redistribute(result, resultDeviceMesh, resultPlacementSeq);
    }
    return result;
}

Tensor _Empty(Graph graph, const IntOrIntArray& shape, const std::optional<DataKind>& dtype,
              const std::optional<DeviceMesh>& deviceMesh, const std::optional<PlacementSeq>& placements) {
    return CreateOpImpl(core::CreateKind::kEmpty, graph, shape, dtype, deviceMesh, placements);
}

Tensor _Zeros(Graph graph, const IntOrIntArray& shape, const std::optional<DataKind>& dtype,
              const std::optional<DeviceMesh>& deviceMesh, const std::optional<PlacementSeq>& placements) {
    return CreateOpImpl(core::CreateKind::kZeros, graph, shape, dtype, deviceMesh, placements);
}

Tensor _Ones(Graph graph, const IntOrIntArray& shape, const std::optional<DataKind>& dtype,
             const std::optional<DeviceMesh>& deviceMesh, const std::optional<PlacementSeq>& placements) {
    return CreateOpImpl(core::CreateKind::kOnes, graph, shape, dtype, deviceMesh, placements);
}

Tensor _Rand(Graph graph, const IntOrIntArray& shape, const std::optional<Generator>& generator,
             const std::optional<DataKind>& dtype, const std::optional<DeviceMesh>& deviceMesh,
             const std::optional<PlacementSeq>& placements) {
    return CreateOpImpl(core::CreateKind::kRand, graph, shape, dtype, deviceMesh, placements, generator);
}

Tensor _Randn(Graph graph, const IntOrIntArray& shape, const std::optional<Generator>& generator,
              const std::optional<DataKind>& dtype, const std::optional<DeviceMesh>& deviceMesh,
              const std::optional<PlacementSeq>& placements) {
    return CreateOpImpl(core::CreateKind::kRandn, graph, shape, dtype, deviceMesh, placements, generator);
}

Tensor _ArangeImp(Graph graph, DataKind dataKind, double arg0, std::optional<double> arg1, std::optional<double> arg2,
                  const std::optional<DataKind>& dtype, const std::optional<DeviceMesh>& deviceMesh,
                  const std::optional<PlacementSeq>& placements) {
    if (!arg1.has_value() && arg2.has_value()) {
        throw std::invalid_argument("Arange operator's end is None, but step has value");
    }

    double start, end, step;
    if (arg1.has_value()) {
        start = arg0;
        end = arg1.value();
        step = arg2.has_value() ? arg2.value() : 1;
    } else {
        start = 0;
        end = arg0;
        step = 1;
    }

    return CreateOpImpl(core::CreateKind::kArange, graph, {}, dtype.value_or(dataKind), deviceMesh, placements,
                        std::nullopt, start, end, step);
}

Tensor _Arange(Graph graph, int64_t start, const std::optional<int64_t>& end, const std::optional<int64_t>& step,
               const std::optional<DataKind>& dtype, const std::optional<DeviceMesh>& deviceMesh,
               const std::optional<PlacementSeq>& placements) {
    return _ArangeImp(graph, DataKind::kInt64, start, end, step, dtype, deviceMesh, placements);
}

Tensor _Arange(Graph graph, double start, const std::optional<double>& end, const std::optional<double>& step,
               const std::optional<DataKind>& dtype, const std::optional<DeviceMesh>& deviceMesh,
               const std::optional<PlacementSeq>& placements) {
    return _ArangeImp(graph, DataKind::kFloat32, start, end, step, dtype, deviceMesh, placements);
}

Tensor _Full(Graph graph, const IntOrIntArray& shape, double fillValue, const std::optional<DataKind>& dtype,
             const std::optional<DeviceMesh>& deviceMesh, const std::optional<PlacementSeq>& placements) {
    return CreateOpImpl(core::CreateKind::kFull, graph, shape, dtype, deviceMesh, placements, std::nullopt, fillValue);
}

Tensor _Randint(Graph graph, int64_t high, const IntOrIntArray& shape, const std::optional<Generator>& generator,
                const std::optional<DataKind>& dtype, const std::optional<DeviceMesh>& deviceMesh,
                const std::optional<PlacementSeq>& placements) {
    return _Randint(graph, 0, high, shape, generator, dtype, deviceMesh, placements);
}

Tensor _Randint(Graph graph, int64_t low, int64_t high, const IntOrIntArray& shape,
                const std::optional<Generator>& generator, const std::optional<DataKind>& dtype,
                const std::optional<DeviceMesh>& deviceMesh, const std::optional<PlacementSeq>& placements) {
    return CreateOpImpl(core::CreateKind::kRandInt, graph, shape, dtype.value_or(DataKind::kInt64), deviceMesh,
                        placements, generator, low, high);
}

Tensor _FromTorch(Graph graph, const torch::Tensor& tensor, const std::optional<DeviceMesh>& deviceMesh,
                  const std::optional<PlacementSeq>& placements) {
    Shape shape = external::torch::TorchUtil::GetShape(tensor);
    DataKind dataKind = external::torch::TorchUtil::GetDataKind(tensor);
    Device tensorDevice = external::torch::TorchUtil::GetDevice(tensor);
    if (tensorDevice.deviceKind == DeviceKind::kGpu) {
        // Synchronize the CUDA stream to wait until the tensor is ready.
        // DTorch will use this tensor in other thread or process.
#if DTORCH_WITH_CUDA
        at::cuda::getCurrentCUDAStream(tensorDevice.deviceId).synchronize();
#endif
    }

    Tensor result =
        CreateOpImpl(core::CreateKind::kFromTorch, graph, shape.Vec(), dataKind,
                     deviceMesh.value_or(DeviceMesh(tensorDevice)), placements, std::nullopt, 0.0, 0.0, 0.0, tensor);

    // If the tensor is a CUDA tensor, we need to sync the graph to wait until the tensor is created.
    // Because the CUDA tensor is created in another process, we need to sync the graph to wait until the tensor is
    // created.
    DDebugAssert(graph.GetGraphOption().perDevicePerProcess.has_value());
    if (graph.GetGraphOption().perDevicePerProcess.value() && tensorDevice.deviceKind == DeviceKind::kGpu) {
        graph.Sync();
    }

    return result;
}

TensorArray _Max(const Tensor& input) {
    std::unique_ptr<core::OpParam> param(new core::MaxMinParam(core::MaxMinKind::kMax));
    return core::GraphConstructor::AddOperator(std::move(param), {input}, true);
}

TensorArray _Max(const Tensor& input, int64_t dim, bool keepdim) {
    std::unique_ptr<core::OpParam> param(new core::MaxMinParam(core::MaxMinKind::kMax, dim, keepdim));
    return core::GraphConstructor::AddOperator(std::move(param), {input}, true);
}

TensorArray _Min(const Tensor& input) {
    std::unique_ptr<core::OpParam> param(new core::MaxMinParam(core::MaxMinKind::kMin));
    return core::GraphConstructor::AddOperator(std::move(param), {input}, true);
}

TensorArray _Min(const Tensor& input, int64_t dim, bool keepdim) {
    std::unique_ptr<core::OpParam> param(new core::MaxMinParam(core::MaxMinKind::kMin, dim, keepdim));
    return core::GraphConstructor::AddOperator(std::move(param), {input}, true);
}

Tensor _Sum(const Tensor& input, const IntOrIntArray& dim, bool keepdim, const std::optional<DataKind>& dtype) {
    std::unique_ptr<core::OpParam> param(new core::ReduceParam(core::ReduceKind::kSum, dim, keepdim, dtype));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Mean(const Tensor& input, const IntOrIntArray& dim, bool keepdim, const std::optional<DataKind>& dtype) {
    std::unique_ptr<core::OpParam> param(new core::ReduceParam(core::ReduceKind::kMean, dim, keepdim, dtype));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Any(const Tensor& input, const IntOrIntArray& dim, bool keepdim) {
    std::unique_ptr<core::OpParam> param(new core::ReduceParam(core::ReduceKind::kAny, dim, keepdim));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _All(const Tensor& input, const IntOrIntArray& dim, bool keepdim) {
    std::unique_ptr<core::OpParam> param(new core::ReduceParam(core::ReduceKind::kAll, dim, keepdim));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Squeeze(const Tensor& input, const std::optional<IntOrIntArray>& dim) {
    std::unique_ptr<core::OpParam> param(new core::SqueezeParam(dim));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Unsqueeze(const Tensor& input, int64_t dim) {
    std::unique_ptr<core::OpParam> param(new core::UnsqueezeParam(dim));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

TensorArray _Chunk(const Tensor& input, int64_t chunk, int64_t dim) {
    std::unique_ptr<core::OpParam> param(new core::ChunkParam(chunk, dim));
    return core::GraphConstructor::AddOperator(std::move(param), {input}, true);
}

Tensor _Concat(const TensorArray& inputs, int64_t dim) {
    DAlwaysAssert(inputs.size() > 0);
    if (inputs.size() == 1) {
        return inputs[0];
    }
    std::unique_ptr<core::OpParam> param(new core::ConcatParam(dim));
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor _Expand(const Tensor& input, const IntOrIntArray& shape) {
    std::unique_ptr<core::OpParam> param(new core::ExpandParam(shape));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Transpose(const Tensor& input, int64_t dim0, int64_t dim1) {
    std::unique_ptr<core::OpParam> param(new core::TransposeParam(dim0, dim1));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Permute(const Tensor& input, const IntOrIntArray& dims) {
    std::unique_ptr<core::OpParam> param(new core::PermuteParam(dims.Vec()));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Repeat(const Tensor& input, const IntOrIntArray& repeats) {
    std::unique_ptr<core::OpParam> param(new core::RepeatParam(repeats.Vec()));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _RepeatInterleave(const Tensor& input, int64_t repeats, const std::optional<int64_t>& dim) {
    std::unique_ptr<core::OpParam> param(new core::RepeatInterleaveParam(repeats, dim));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor Pad(const Tensor& input, const IntOrIntArray& pad, const std::string& mode, const std::optional<double>& value) {
    if (pad.size() == 0) {
        return input;
    }

    std::unique_ptr<core::OpParam> param(new core::PadParam(pad.Vec(), mode, value));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _GetItem(const Tensor& input, const std::vector<Index>& indexs) {
    std::unique_ptr<core::OpParam> param(new core::GetItemParam(indexs));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _SetItem(const Tensor& input, const Tensor& value, const std::vector<Index>& indexs) {
    std::unique_ptr<core::OpParam> param(new core::SetItemParam(indexs));
    return core::GraphConstructor::AddOperator(std::move(param), {input, value});
}

void _Copy(Tensor input, const Tensor& other) {
    Tensor formatedTensor = other;
    if (input.IsDistributed() && !other.IsDistributed()) {
        formatedTensor = _Redistribute(other, input.GetDeviceMesh(), input.GetPlacementSeq().Vec());
    } else if (!input.IsDistributed() && other.IsDistributed()) {
        DeviceMesh deviceMesh(Device(other.GetDeviceKind()));
        PlacementSeq placements = {Replicate()};
        formatedTensor = _Redistribute(other, deviceMesh, placements);
    } else if (input.IsDistributed() && other.IsDistributed()) {
        formatedTensor = _Redistribute(other, input.GetDeviceMesh(), input.GetPlacementSeq().Vec());
    }

    std::unique_ptr<core::CopyParam> param(new core::CopyParam());
    TensorArray array = core::GraphConstructor::AddOperator(std::move(param), {input, formatedTensor}, true);
    DDebugAssert(array.size() == 0);
}

Tensor _View(const Tensor& input, const IntOrIntArray& shape) {
    std::unique_ptr<core::OpParam> param(new core::ViewParam(shape));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _View(const Tensor& input, const PlacementSeq& placements) {
    std::unique_ptr<core::OpParam> param(new core::ViewParam(placements));
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Contiguous(const Tensor& input) {
    std::unique_ptr<core::OpParam> param(new core::ContiguousParam());
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _Clone(const Tensor& input) {
    std::unique_ptr<core::OpParam> param(new core::CloneParam());
    return core::GraphConstructor::AddOperator(std::move(param), {input});
}

Tensor _To(const Tensor& input, const std::optional<DeviceMesh>& device_mesh, const std::optional<DataKind>& dtype) {
    if (device_mesh.has_value() && device_mesh.value().GetMeshShape() != input.GetDeviceMesh().GetMeshShape()) {
        std::stringstream ss;
        ss << "The device_mesh in the parameter and the device_mesh of the input tensor have different shapes, "
           << device_mesh.value().GetMeshShape() << " vs " << input.GetDeviceMesh().GetMeshShape() << ". "
           << "If you need to convert the device_mesh of a tensor, use the tensor.redistribute() instead.";
        throw std::invalid_argument("");
    }

    return ConvertOpImpl::Call(input, dtype, device_mesh, std::nullopt);
}

Tensor _Redistribute(const Tensor& input, const std::optional<DeviceMesh>& deviceMesh,
                     const std::optional<PlacementSeq>& placementSeq) {
    return ConvertOpImpl::Call(input, std::nullopt, deviceMesh, placementSeq);
}

Tensor _Clamp(const Tensor& input, const std::optional<double>& min, const std::optional<double>& max) {
    std::unique_ptr<core::OpParam> param(new core::ClampParam(min, max));
    TensorArray inputs = {input};
    TensorArrayPushOptional(inputs);
    TensorArrayPushOptional(inputs);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor _Clamp(const Tensor& input, const std::optional<Tensor>& min, const std::optional<Tensor>& max) {
    std::unique_ptr<core::OpParam> param(new core::ClampParam());
    TensorArray inputs = {input};
    TensorArrayPushOptional(inputs, min);
    TensorArrayPushOptional(inputs, max);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor _WhereImp(const Tensor& condition, const std::optional<Tensor>& input, const std::optional<Tensor>& other,
                 std::optional<double> inputScalar, std::optional<double> otherScalar) {
    std::unique_ptr<core::OpParam> param(new core::WhereParam(inputScalar, otherScalar));
    TensorArray inputs = {condition};
    TensorArrayPushOptional(inputs, input);
    TensorArrayPushOptional(inputs, other);
    return core::GraphConstructor::AddOperator(std::move(param), inputs);
}

Tensor _Where(const Tensor& condition, const Tensor& input, const Tensor& other) {
    return _WhereImp(condition, input, other, std::nullopt, std::nullopt);
}

Tensor _Where(const Tensor& condition, const Scalar& input, const Tensor& other) {
    return _WhereImp(condition, std::nullopt, other, input.Value<double>(), std::nullopt);
}

Tensor _Where(const Tensor& condition, const Tensor& input, const Scalar& other) {
    return _WhereImp(condition, input, std::nullopt, std::nullopt, other.Value<double>());
}

Tensor _Where(const Tensor& condition, const Scalar& input, const Scalar& other) {
    return _WhereImp(condition, std::nullopt, std::nullopt, input.Value<double>(), other.Value<double>());
}

Tensor _MaskedFill(const Tensor& input, const Tensor& mask, const Scalar& value) {
    std::unique_ptr<core::OpParam> param(new core::MaskedParam(core::MaskedType::kMaskedFill, value.Value<double>()));
    return core::GraphConstructor::AddOperator(std::move(param), {input, mask});
}

Tensor _MaskedScatter(const Tensor& input, const Tensor& mask, const Tensor& source) {
    std::unique_ptr<core::OpParam> param(new core::MaskedParam(core::MaskedType::kMaskedScatter));
    return core::GraphConstructor::AddOperator(std::move(param), {input, mask, source});
}

TensorFuture _GetTensorAsync(const Tensor& input) {
    // 1. If input is DTensor, redistribute to local tensor first
    //    (GetTensorOp can only handle local tensors)
    Tensor localTensor = input;
    if (input.IsDistributed()) {
        DeviceMesh localDevice(Device(input.GetDeviceKind(), 0));
        localTensor = _Redistribute(input, localDevice);
    }

    // 2. Get Operand
    auto& operand = localTensor.GetOperand();

    // 3. Determine Promise type — use the per-graph setting, not GlobalOption singleton
    bool perDevicePerProcess = input.GetGraph().GetGraphOption().perDevicePerProcess.value_or(false);
    auto promiseType = core::communication::GetTensorPromiseTypeFromOperand(*operand, perDevicePerProcess);

    // 4. Create Promise, get Future
    auto promise = core::communication::CreateTensorPromise(promiseType);
    auto future = promise->GetFuture();

    // 5. Create GetTensorParam (holds promise)
    auto param = std::make_unique<core::GetTensorParam>(std::move(promise));

    // 6. Create GetTensorOp and add to compute graph
    //    GetTensorOp has 0 outputs, so use the returnArray=true overload
    core::GraphConstructor::AddOperator(std::move(param), {localTensor}, true);

    // 7. Return TensorFuture
    return TensorFuture(input, std::move(future));
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
