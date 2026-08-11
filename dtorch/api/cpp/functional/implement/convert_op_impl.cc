/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "convert_op_impl.h"

#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/common/debug.h"
#include "dtorch/core/distributed/cluster_info.h"
#include "dtorch/core/operators/standard/convert_op.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

Tensor ConvertOpImpl::RunFunc() {
    if (mInputIsDistributed && mOutputIsDistributed && mSrcDeviceMesh.NumAxis() != mDestDeviceMesh.NumAxis()) {
        std::stringstream ss;
        ss << "Not support convert " << mSrcDeviceMesh.NumAxis() << "D DTensor to " << mDestDeviceMesh.NumAxis()
           << "D DTensor";
        throw std::invalid_argument(ss.str());
    }

    mDestPlacements.ToReplicateWhenDimSizeEqualOne(mDestDeviceMesh);
    mDestPlacements.ShardDimToPositiveNumber(mInput.GetShape());
    ConvertPlacementsOneByOne();
    return AddIntoGraphConstructor(mInput, mDestDataKind, mDestDeviceMesh, mDestPlacements);
}

Tensor ConvertOpImpl::AddIntoGraphConstructor(const Tensor& input, DataKind dataKind, const DeviceMesh& deviceMesh,
                                              const PlacementSeq& placements) {
    if (!mConvertDataKind && !mConvertDeviceMesh && !mConvertPlacement) {
        return input;
    }

    int64_t totalGpuCount = core::distributed::ClusterInfo::GetSingleton().GetTotalGpuCount();
    if (!DistributedSpec::CheckValid(mInput.GetShape(), deviceMesh, placements, totalGpuCount)) {
        std::stringstream ss;
        ss << "shape, device_mesh and placements is invalid, shape: " << mInput.GetShape()
           << ", device_mesh: " << deviceMesh << ", placements: " << placements;
        throw std::invalid_argument(ss.str());
    }

    std::unique_ptr<core::OpParam> param(new core::ConvertParam(dataKind, deviceMesh, placements));
    return core::GraphConstructor::AddOperator(std::move(param), {mInput});
}

void ConvertOpImpl::ConvertPlacementsOneByOne() {
    if (!mConvertPlacement || !mInputIsDistributed || !mOutputIsDistributed) {
        return;
    }

    DDebugAssert(mSrcPlacements.Size() == mDestPlacements.Size());
    auto convertOrder = mSrcPlacements.GetConvertOrder(mDestPlacements);
    PlacementSeq convertPlacement = mSrcPlacements;
    for (auto diffDim : convertOrder) {
        convertPlacement[diffDim] = mDestPlacements[diffDim];
        mInput = AddIntoGraphConstructor(mInput, mSrcDataKind, mSrcDeviceMesh, convertPlacement);
    }

    mSrcPlacements = mDestPlacements;
    mConvertPlacement = false;
}

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
