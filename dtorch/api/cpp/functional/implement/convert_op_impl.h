/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "dtorch/core/graph/graph_constructor.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

class ConvertOpImpl {
public:
    static Tensor Call(const Tensor& input, const std::optional<DataKind>& dataKind,
                       const std::optional<DeviceMesh>& deviceMesh, const std::optional<PlacementSeq>& placements) {
        ConvertOpImpl funcUtil(input, dataKind, deviceMesh, placements);
        return funcUtil.RunFunc();
    }

private:
    ConvertOpImpl(const Tensor& input, const std::optional<DataKind>& dataKind,
                  const std::optional<DeviceMesh>& deviceMesh, const std::optional<PlacementSeq>& placements)
        : mInput(input),
          mSrcDataKind(mInput.GetDataKind()),
          mSrcDeviceMesh(mInput.GetDeviceMesh()),
          mSrcPlacements(mInput.GetPlacementSeq()),
          mDestDataKind(dataKind.value_or(mSrcDataKind)),
          mDestDeviceMesh(deviceMesh.value_or(mSrcDeviceMesh)),
          mDestPlacements(),
          mInputIsDistributed(mInput.IsDistributed()),
          mOutputIsDistributed(mDestDeviceMesh.IsDistributed()),
          mConvertDataKind(mSrcDataKind != mDestDataKind),
          mConvertDeviceMesh(mSrcDeviceMesh != mDestDeviceMesh),
          mConvertPlacement(mSrcPlacements != mDestPlacements) {
        if (deviceMesh.has_value()) {
            mDestPlacements = placements.value_or(PlacementSeq(deviceMesh.value().NumAxis(), Replicate()));
        } else {
            mDestPlacements = placements.value_or(mSrcPlacements);
        }
    }

    Tensor RunFunc();

    Tensor AddIntoGraphConstructor(const Tensor& input, DataKind dataKind, const DeviceMesh& deviceMesh,
                                   const PlacementSeq& placements);

    void ConvertPlacementsOneByOne();

private:
    Tensor mInput;
    DataKind mSrcDataKind;
    DeviceMesh mSrcDeviceMesh;
    PlacementSeq mSrcPlacements;
    DataKind mDestDataKind;
    DeviceMesh mDestDeviceMesh;
    PlacementSeq mDestPlacements;
    bool mInputIsDistributed;
    bool mOutputIsDistributed;
    bool mConvertDataKind;
    bool mConvertDeviceMesh;
    bool mConvertPlacement;
};

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
