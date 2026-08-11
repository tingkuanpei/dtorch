/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "dtorch/core/operand.h"

#include "dtorch/common/debug.h"

namespace dtorch {
namespace core {

Operand::Operand()
    : mName(""),
      mProducerOp(nullptr),
      mConsumerOps(),
      mShape(),
      mStride(),
      mDataKind(DataKind::kFloat32),
      mDeviceMesh(Device::GetDefaultCpuDevice()),
      mPlacementSeq({Placement("R")}),
      mMutex(),
      mAllLocalShape() {}

Operand::Operand(const Shape& shape, DataKind dataKind, const DeviceMesh& mesh, const PlacementSeq& placementSeq)
    : mName(""),
      mProducerOp(nullptr),
      mConsumerOps(),
      mShape(shape),
      mStride(shape),
      mDataKind(dataKind),
      mDeviceMesh(mesh),
      mPlacementSeq(placementSeq),
      mMutex(),
      mAllLocalShape() {}

Operand::~Operand() {}

std::string Operand::GetTypeString() const {
    std::stringstream ss;
    if (IsDistributed()) {
        ss << "Distribute";
    }

    std::string result = ss.str();
    if (result.empty()) {
        result = "Normal";
    }

    return result;
}

//---------------------------------------------- Meta Info -------------------------------------------------------------

void Operand::SetShapeAndStride(const Shape& shape, const Stride& stride) {
    mShape = shape;
    if (stride.Empty()) {
        mStride = Stride(shape);
    } else {
        mStride = stride;
    }
    mAllLocalShape.clear();
}

void Operand::MetaDataSameAs(const Operand* otherPtr) {
    this->mShape = otherPtr->mShape;
    this->mStride = otherPtr->mStride;
    this->mDataKind = otherPtr->mDataKind;
    this->mDeviceMesh = otherPtr->mDeviceMesh;
    this->mPlacementSeq = otherPtr->mPlacementSeq;
    this->mAllLocalShape.clear();
}

//--------------------------------------------- Distribute -------------------------------------------------------------

Shape Operand::GetLocalShape(int64_t globalDeviceId) const noexcept {
    ComputeLocalShape();

    DDebugAssert(mAllLocalShape.size() > 0);
    if (mAllLocalShape.size() == 1) {
        return mAllLocalShape.begin()->second;
    } else {
        DDebugAssert(GetDistributedDeviceIdSet().count(globalDeviceId) > 0);
        DDebugAssert(GetDistributedDeviceIdSet().size() == mAllLocalShape.size());
        auto it = mAllLocalShape.find(globalDeviceId);
        DDebugAssert(it != mAllLocalShape.end());
        return it->second;
    }
}

bool Operand::IsLocalShapeEventSplit() const noexcept {
    ComputeLocalShape();
    DDebugAssert(mAllLocalShape.size() > 0);
    return mAllLocalShape.size() == 1;
}

void Operand::SetDeviceMeshAndPlacementSeq(const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq) {
    mDeviceMesh = deviceMesh;
    mPlacementSeq = placementSeq;
    mAllLocalShape.clear();
}

const std::unordered_set<int64_t>& Operand::GetDistributedDeviceIdSet() const {
    return GetDeviceMesh().GetDeviceIdSet();
}

void Operand::ComputeLocalShape() const {
    std::unique_lock<std::mutex> guard(mMutex);
    if (mAllLocalShape.size() == 0) {
        if (IsDistributed()) {
            mAllLocalShape = DistributedSpec::ComputeLocalShape(mShape, mDeviceMesh, mPlacementSeq);
        } else {
            mAllLocalShape.clear();
            mAllLocalShape[0] = mShape;
        }
    }
}

}  // namespace core
}  // namespace dtorch
