/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "convert_op.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

void ConvertOp::InferOutputMetaInfo() const {
    DDebugAssert(GetInputSize() == 1);
    const auto& param = GetOpParam<ConvertParam>();

    Operand* out = OperandY();
    out->MetaDataSameAs(OperandX());
    out->SetDataKind(param.dataKind);
    out->SetDeviceMeshAndPlacementSeq(param.deviceMesh, param.placementSeq);
}

std::string ConvertOp::GetDescribeString() const {
    DDebugAssert(GetInputSize() >= 1);
    const auto& param = GetOpParam<ConvertParam>();

    std::stringstream ss;
    ss << GetOpType() << ": ";
    if (OperandX()->GetDataKind() != param.dataKind) {
        ss << "Datakind " << OperandX()->GetDataKind() << " -> " << param.dataKind << ", ";
    }
    if (OperandX()->GetDeviceMesh() != param.deviceMesh) {
        ss << "DeviceMesh " << OperandX()->GetDeviceMesh() << " -> " << param.deviceMesh << ", ";
    }
    if (OperandX()->GetPlacementSeq() != param.placementSeq) {
        ss << "Placements " << OperandX()->GetPlacementSeq() << " -> " << param.placementSeq << ", ";
    }
    return ss.str();
}

}  // namespace core
}  // namespace dtorch
