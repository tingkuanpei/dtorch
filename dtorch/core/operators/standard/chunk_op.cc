/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "chunk_op.h"

#include <csignal>
#include <stdexcept>

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

size_t ChunkOp::InferOutputSize() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inShape = OperandX()->GetShape();
    const auto& param = GetOpParam<ChunkParam>();
    size_t dim = Operator::GetValidDim(inShape, param.dim);
    if (inShape[dim] <= param.chunks) {
        return inShape[dim];
    } else {
        size_t perOutputSize = (inShape[dim] + param.chunks - 1) / param.chunks;
        size_t outputSize = (inShape[dim] + perOutputSize - 1) / perOutputSize;
        return outputSize;
    }
}

void ChunkOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const Shape& inShape = OperandX()->GetShape();
    const auto& param = GetOpParam<ChunkParam>();
    Shape outShape = inShape;
    Shape lastTensorOutShape = inShape;

    size_t dim = Operator::GetValidDim(inShape, param.dim);
    if (inShape[dim] <= param.chunks) {
        outShape[dim] = 1;
        lastTensorOutShape[dim] = 1;
    } else {
        outShape[dim] = (inShape[dim] + param.chunks - 1) / param.chunks;
        lastTensorOutShape[dim] = inShape[dim] - outShape[dim] * (GetOutputSize() - 1);
    }

    for (size_t i = 0; i < GetOutputSize() - 1; i++) {
        GetOutputOperand(i)->MetaDataSameAs(OperandX());
        GetOutputOperand(i)->SetShapeAndStride(outShape);
    }
    GetOutputOperand(GetOutputSize() - 1)->MetaDataSameAs(OperandX());
    GetOutputOperand(GetOutputSize() - 1)->SetShapeAndStride(lastTensorOutShape);

    // Check placements
    const auto& placements = OperandX()->GetPlacementSeq();
    for (size_t i = 0; i < placements.Size(); i++) {
        if (placements[i].IsShard(static_cast<int>(dim))) {
            throw std::invalid_argument("Not support shard in chunk dim");
        }
    }
}

}  // namespace core
}  // namespace dtorch
