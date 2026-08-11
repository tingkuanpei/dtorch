/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "util.h"

#include <sstream>

#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/functional/tensor_functional.h"
#include "dtorch/common/debug.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor PlacementR2S(const Tensor& replicateTensor, const Tensor& shardTensor) {
    if (!replicateTensor.IsDistributed() || !shardTensor.IsDistributed()) {
        return replicateTensor;
    }

    PlacementSeq placementsA = replicateTensor.GetPlacementSeq();
    PlacementSeq placementsB = shardTensor.GetPlacementSeq();
    if (placementsA.Size() != placementsB.Size()) {
        std::stringstream ss;
        ss << "Placements have different size: " << placementsA.Size() << " vs " << placementsB.Size();
        throw std::invalid_argument(ss.str());
    }

    // TODO: support sub-shard
    for (size_t i = 0; i < placementsA.Size(); i++) {
        if (placementsA[i].IsReplicate() && placementsB[i].IsShard()) {
            Shape inputAShape = replicateTensor.GetShape();
            Shape inputBShape = shardTensor.GetShape();
            int64_t shardIdxInB = placementsB[i].GetShardIndex();
            int64_t subSplitCoordinatesInB = placementsB[i].GetSubSplitCoordinates();
            int64_t shardIdxInA = shardIdxInB - inputBShape.NumAxis() + inputAShape.NumAxis();
            if (shardIdxInA < 0 || inputAShape[shardIdxInA] <= 1) {
                continue;
            }
            DDebugAssert(shardIdxInA < static_cast<int64_t>(inputAShape.NumAxis()));
            placementsA[i] = Shard(shardIdxInA, subSplitCoordinatesInB);
        }
    }

    return _Redistribute(replicateTensor, std::nullopt, placementsA);
};

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
