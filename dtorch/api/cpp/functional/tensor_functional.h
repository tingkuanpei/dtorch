/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <optional>

#include "../scalar.h"
#include "../tensor.h"
#include "../tensor_future.h"
#include "dtorch/api/cpp/data_kind.h"
#include "dtorch/api/cpp/generator.h"
#include "dtorch/api/cpp/index.h"
#include "dtorch/api/cpp/shape.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor _Empty(Graph graph, const IntOrIntArray& shape, const std::optional<DataKind>& dtype = std::nullopt,
              const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
              const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Zeros(Graph graph, const IntOrIntArray& shape, const std::optional<DataKind>& dtype = std::nullopt,
              const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
              const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Ones(Graph graph, const IntOrIntArray& shape, const std::optional<DataKind>& dtype = std::nullopt,
             const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
             const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Rand(Graph graph, const IntOrIntArray& shape, const std::optional<Generator>& generator = std::nullopt,
             const std::optional<DataKind>& dtype = std::nullopt,
             const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
             const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Randn(Graph graph, const IntOrIntArray& shape, const std::optional<Generator>& generator = std::nullopt,
              const std::optional<DataKind>& dtype = std::nullopt,
              const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
              const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Arange(Graph graph, int64_t start, const std::optional<int64_t>& end = std::nullopt,
               const std::optional<int64_t>& step = std::nullopt, const std::optional<DataKind>& dtype = std::nullopt,
               const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
               const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Arange(Graph graph, double start, const std::optional<double>& end = std::nullopt,
               const std::optional<double>& step = std::nullopt, const std::optional<DataKind>& dtype = std::nullopt,
               const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
               const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Full(Graph graph, const IntOrIntArray& shape, double fillValue,
             const std::optional<DataKind>& dtype = std::nullopt,
             const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
             const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Randint(Graph graph, int64_t high, const IntOrIntArray& shape,
                const std::optional<Generator>& generator = std::nullopt,
                const std::optional<DataKind>& dtype = std::nullopt,
                const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
                const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Randint(Graph graph, int64_t low, int64_t high, const IntOrIntArray& shape,
                const std::optional<Generator>& generator = std::nullopt,
                const std::optional<DataKind>& dtype = std::nullopt,
                const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
                const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _FromTorch(Graph graph, const torch::Tensor& tensor, const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
                  const std::optional<PlacementSeq>& placements = std::nullopt);

TensorArray _Max(const Tensor& input);

TensorArray _Max(const Tensor& input, int64_t dim, bool keepdim = false);

TensorArray _Min(const Tensor& input);

TensorArray _Min(const Tensor& input, int64_t dim, bool keepdim = false);

Tensor _Sum(const Tensor& input, const IntOrIntArray& dim = {}, bool keepdim = false,
            const std::optional<DataKind>& dtype = std::nullopt);

Tensor _Mean(const Tensor& input, const IntOrIntArray& dim = {}, bool keepdim = false,
             const std::optional<DataKind>& dtype = std::nullopt);

Tensor _Any(const Tensor& input, const IntOrIntArray& dim = {}, bool keepdim = false);

Tensor _All(const Tensor& input, const IntOrIntArray& dim = {}, bool keepdim = false);

Tensor _Squeeze(const Tensor& input, const std::optional<IntOrIntArray>& dim = std::nullopt);

Tensor _Unsqueeze(const Tensor& input, int64_t dim);

TensorArray _Chunk(const Tensor& input, int64_t chunk, int64_t dim);

Tensor _Concat(const TensorArray& inputs, int64_t dim = 0);

Tensor _Expand(const Tensor& input, const IntOrIntArray& shape);

Tensor _Transpose(const Tensor& input, int64_t dim0, int64_t dim1);

Tensor _Permute(const Tensor& input, const IntOrIntArray& dims);

Tensor _Repeat(const Tensor& input, const IntOrIntArray& repeats);

Tensor _RepeatInterleave(const Tensor& input, int64_t repeats, const std::optional<int64_t>& dim = std::nullopt);

Tensor Pad(const Tensor& input, const IntOrIntArray& pad, const std::string& mode = "constant",
           const std::optional<double>& value = std::nullopt);

Tensor _GetItem(const Tensor& input, const std::vector<Index>& indexs);

Tensor _SetItem(const Tensor& input, const Tensor& value, const std::vector<Index>& indexs);

void _Copy(Tensor input, const Tensor& other);

Tensor _View(const Tensor& input, const IntOrIntArray& shape);

Tensor _View(const Tensor& input, const PlacementSeq& placements);

Tensor _Contiguous(const Tensor& input);

Tensor _Clone(const Tensor& input);

Tensor _To(const Tensor& input, const std::optional<DeviceMesh>& device_mesh = std::nullopt,
           const std::optional<DataKind>& dtype = std::nullopt);

Tensor _Redistribute(const Tensor& input, const std::optional<DeviceMesh>& deviceMesh = std::nullopt,
                     const std::optional<PlacementSeq>& placements = std::nullopt);

Tensor _Clamp(const Tensor& input, const std::optional<double>& min = std::nullopt,
              const std::optional<double>& max = std::nullopt);

Tensor _Clamp(const Tensor& input, const std::optional<Tensor>& min = std::nullopt,
              const std::optional<Tensor>& max = std::nullopt);

Tensor _Where(const Tensor& condition, const Tensor& input, const Tensor& other);

Tensor _Where(const Tensor& condition, const Scalar& input, const Tensor& other);

Tensor _Where(const Tensor& condition, const Tensor& input, const Scalar& other);

Tensor _Where(const Tensor& condition, const Scalar& input, const Scalar& other);

Tensor _MaskedFill(const Tensor& input, const Tensor& mask, const Scalar& value);

Tensor _MaskedScatter(const Tensor& input, const Tensor& mask, const Tensor& source);

// Async get tensor value, returns TensorFuture immediately
TensorFuture _GetTensorAsync(const Tensor& input);

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
