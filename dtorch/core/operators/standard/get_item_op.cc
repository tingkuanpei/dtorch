/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "get_item_op.h"

#include <memory>

#include <ATen/Tensor.h>

#include "dtorch/api/cpp/shape.h"
#include "dtorch/common/debug.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {

Shape GetItemOp::ComputeIndexedShape(const Shape& inShape, std::vector<Index>& indexVec) {
    // First pass: compute broadcast shape of all tensor indices
    Shape broadcastTensorShape;
    size_t numTensorIndices = 0;
    for (size_t i = 0; i < indexVec.size(); i++) {
        if (indexVec[i].IsTensor()) {
            numTensorIndices++;
            const at::Tensor& tensor = indexVec[i].GetTensor();
            Shape tensorShape(tensor.sizes().vec());
            if (broadcastTensorShape.NumAxis() == 0) {
                broadcastTensorShape = tensorShape;
            } else {
                broadcastTensorShape = Shape::BroadcastOutputShape(broadcastTensorShape, tensorShape);
            }
        }
    }

    int64_t numOfEllipsis = 0;
    int64_t numOfNone = 0;
    for (size_t i = 0; i < indexVec.size(); i++) {
        if (indexVec[i].IsEllipsis()) {
            numOfEllipsis++;
        }
        if (indexVec[i].IsNone()) {
            numOfNone++;
        }
    }

    if (numOfEllipsis >= 3 || (numOfEllipsis == 2 && !indexVec.back().IsEllipsis())) {
        throw std::invalid_argument("Index op InferOutputShape error, get too many ellipsis");
    }
    if (numOfEllipsis == 2) {
        DDebugAssert(indexVec.back().IsEllipsis());
        indexVec.pop_back();
        numOfEllipsis = 1;
    }

    if ((inShape.NumAxis() + numOfNone) < (indexVec.size() - numOfEllipsis)) {
        throw std::invalid_argument("Index op InferOutputShape error, get too many index");
    }

    int64_t rangeEllipsis = -1;
    if (numOfEllipsis >= 1) {
        rangeEllipsis = inShape.NumAxis() + numOfNone - indexVec.size() + 1;
    }

    auto CheckInteger = [](Shape::DataType dimSize, int64_t index) {
        if (index < -1 * dimSize || index >= dimSize) {
            throw std::invalid_argument("Index op InferOutputShape error, index out of range");
        }
    };

    auto CheckSlice = [](Shape::DataType dimSize, Slice slice) {
        int64_t start = 0;
        int64_t stop = dimSize;
        int64_t step = 1;

        if (slice.start.has_value()) {
            auto tmpValue = slice.start.value();
            start = tmpValue < 0 ? tmpValue + dimSize : tmpValue;

            start = start < 0 ? 0 : start;
            start = start > dimSize ? dimSize : start;
        }
        if (slice.stop.has_value()) {
            auto tmpValue = slice.stop.value();
            stop = tmpValue < 0 ? tmpValue + dimSize : tmpValue;

            stop = stop < 0 ? 0 : stop;
            stop = stop > dimSize ? dimSize : stop;
        }
        if (slice.step.has_value()) {
            step = slice.step.value();
        }

        int64_t result = (stop - start - 1) / step + 1;
        return result < 0 ? 0 : result;
    };

    // Build intermediate shape treating tensor indices like integer indices (consume dim, produce no output)
    Shape outShape;
    size_t inIdx = 0;

    for (size_t indexIdx = 0; indexIdx < indexVec.size(); indexIdx++) {
        if (indexVec[indexIdx].IsInteger()) {
            CheckInteger(inShape[inIdx], indexVec[indexIdx].GetInteger());
            inIdx++;
        } else if (indexVec[indexIdx].IsTensor()) {
            // Tensor index: consumes one input dimension (replaced by broadcast shape later)
            inIdx++;
        } else if (indexVec[indexIdx].IsNone()) {
            outShape.PushBack(1);
        } else if (indexVec[indexIdx].IsEllipsis()) {
            for (int64_t i = 0; i < rangeEllipsis; i++) {
                outShape.PushBack(inShape[inIdx]);
                inIdx++;
            }
        } else if (indexVec[indexIdx].IsSlice()) {
            outShape.PushBack(CheckSlice(inShape[inIdx], indexVec[indexIdx].GetSlice()));
            inIdx++;
        }
    }

    for (; inIdx < inShape.NumAxis(); inIdx++) {
        outShape.PushBack(inShape[inIdx]);
    }

    // Prepend broadcast tensor shape if there are tensor indices
    if (numTensorIndices > 0) {
        Shape result = broadcastTensorShape;
        for (size_t i = 0; i < outShape.NumAxis(); i++) {
            result.PushBack(outShape[i]);
        }
        return result;
    }

    return outShape;
}

void GetItemOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    auto indexVec = GetOpParam<GetItemParam>().indexVec;
    const Shape& inShape = OperandX()->GetShape();

    Shape outShape = ComputeIndexedShape(inShape, indexVec);

    // Build placement signature for distributed tensors
    if (OperandX()->IsDistributed()) {
        mPlacementSignatureBuilder = std::make_unique<PlacementSignature::Builder>(GetInputSize(), GetOutputSize());

        // Re-derive rangeEllipsis from the normalized indexVec
        int64_t numOfNone = 0;
        int64_t numOfEllipsis = 0;
        for (const auto& idx : indexVec) {
            if (idx.IsNone()) numOfNone++;
            if (idx.IsEllipsis()) numOfEllipsis++;
        }
        int64_t rangeEllipsis = -1;
        if (numOfEllipsis >= 1) {
            rangeEllipsis = inShape.NumAxis() + numOfNone - indexVec.size() + 1;
        }

        size_t inIdx = 0;
        size_t outIdx = 0;

        for (size_t indexIdx = 0; indexIdx < indexVec.size(); indexIdx++) {
            if (indexVec[indexIdx].IsInteger() || indexVec[indexIdx].IsTensor()) {
                inIdx++;
            } else if (indexVec[indexIdx].IsNone()) {
                outIdx++;
            } else if (indexVec[indexIdx].IsEllipsis()) {
                for (int64_t i = 0; i < rangeEllipsis; i++) {
                    mPlacementSignatureBuilder->AddInput(Shard(inIdx)).AddOutput(Shard(outIdx)).Build();
                    outIdx++;
                    inIdx++;
                }
            } else if (indexVec[indexIdx].IsSlice()) {
                Slice& slice = const_cast<Slice&>(indexVec[indexIdx].GetSlice());
                bool supportShard = false;
                if (inShape[inIdx] == outShape[outIdx]) {
                    supportShard = true;
                }

                size_t shardSize = 0;
                std::vector<DistributedSpec::ShardInfo> shardInfo =
                    DistributedSpec::GetShardInfo(OperandX()->GetDeviceMesh(), OperandX()->GetPlacementSeq());
                for (const auto& it : shardInfo) {
                    if (it.shardDim == inIdx) {
                        shardSize = it.shardSize;
                        break;
                    }
                }
                // TODO: dirty code!!! when encountering bugs, please rewrite this part of the code.
                const PlacementSeq& placements = OperandX()->GetPlacementSeq();
                for (size_t i = 0; i < placements.Size(); i++) {
                    if (placements[i].HasSubSplitCoordinates()) {
                        int64_t subSplitCoordinates = placements[i].GetSubSplitCoordinates();
                        if (!slice.start.has_value() && !slice.step.has_value() && slice.stop.has_value() &&
                            slice.stop.value() == subSplitCoordinates + 1) {
                            supportShard = true;
                            mKeepSubSplitCoordinates = false;
                            if (shardSize != 0) {
                                slice.stop = slice.stop.value() / shardSize;
                            }
                        }
                        if (!slice.stop.has_value() && !slice.step.has_value() && slice.start.has_value() &&
                            slice.start.value() == subSplitCoordinates + 1) {
                            supportShard = true;
                            mKeepSubSplitCoordinates = false;
                            if (shardSize != 0) {
                                slice.start = slice.start.value() / shardSize;
                            }
                        }
                    }
                }

                if (supportShard) {
                    mPlacementSignatureBuilder->AddInput(Shard(inIdx)).AddOutput(Shard(outIdx)).Build();
                }
                inIdx++;
                outIdx++;
            }
        }

        for (; inIdx < inShape.NumAxis(); inIdx++) {
            mPlacementSignatureBuilder->AddInput(Shard(inIdx)).AddOutput(Shard(outIdx)).Build();
            outIdx++;
        }
    }

    OperandY()->MetaDataSameAs(OperandX());
    OperandY()->SetShapeAndStride(outShape);

    mIndexVec = indexVec;
}

PlacementSignature GetItemOp::GetPlacementSignature() const {
    DDebugAssert(GetInputSize() == 1);
    mPlacementSignatureBuilder->AddInput("P").AddOutput("P").Build();
    return mPlacementSignatureBuilder->Finish();
}

void GetItemOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(inputs.size() == 1);
    // PyTorch's make_info() in TensorAdvancedIndexingUtils.h internally calls
    // indice.to(self.device()) when the index tensor is on CPU but self is on CUDA.
    // This triggers a synchronous cudaMemcpyAsync (pageable memory → staging buffer),
    // causing host-thread stall visible in Nsight Systems as a CUDA sync.
    // Ref: https://github.com/pytorch/pytorch/blob/release/2.8/aten/src/ATen/native/TensorAdvancedIndexingUtils.h#L97
    // We move CPU index tensors to the target device ahead of time with non_blocking=true
    // to avoid the implicit sync inside make_info().
    const auto& device = inputs[0].value().device();
    for (auto& index : mIndexVec) {
        if (index.IsTensor()) {
            auto& tensor = const_cast<at::Tensor&>(index.GetTensor());
            // tensor device MUST: 1. cpu 2. same as inputs[0].value()
            if (tensor.device() != device) {
                tensor = tensor.to(device, /*non_blocking=*/true);
            }
        }
    }
    outputs.push_back(inputs[0].value().index(external::torch::TorchUtil::ToIndex(mIndexVec)));
}

}  // namespace core
}  // namespace dtorch
