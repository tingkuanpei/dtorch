/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include "../operator.h"

namespace dtorch {
namespace core {

// Convert a Tensor from one format to another. Only the following conversions are implemented here; other format
// conversions are handled in ConvertOpImpl.
// 1. Local Tensor conversions: 1. Device 2. DataKind 3. Both device and dataKind
// 2. DTensor conversions: 1. DataKind 2. DeviceMesh with the same shape 3. Placements for one dimension
// 3. Convert a local tensor to a DTensor
// 4. Convert a DTensor to a local tensor

struct ConvertParam : public OpParam {
    DataKind dataKind;
    DeviceMesh deviceMesh;
    PlacementSeq placementSeq;

public:
    ConvertParam() : OpParam(OperatorType::kConvert), dataKind(DataKind::kFloat32), deviceMesh(), placementSeq() {}

    ConvertParam(DataKind dataKind, DeviceMesh deviceMesh, PlacementSeq placementSeq)
        : OpParam(OperatorType::kConvert), dataKind(dataKind), deviceMesh(deviceMesh), placementSeq(placementSeq) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar& BaseObject<OpParam>(*this);
        ar & dataKind;
        ar & deviceMesh;
        ar & placementSeq;
    }
};

class ConvertOp : public Operator {
public:
public:
    ConvertOp(std::shared_ptr<OpParam> opParamPtr) : Operator(opParamPtr) {}

    void InferOutputMetaInfo() const override;

    bool SkipDistributedSpecFromPlacementSignature() const override { return true; }

    std::string GetDescribeString() const override;
};

}  // namespace core
}  // namespace dtorch
