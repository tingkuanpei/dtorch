/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "dtorch/core/type.h"

namespace dtorch {
namespace core {

class Operator;
class LogicalGraph;

// 目前涉及 inplace 操作的算子：ActivationOp、SetItemOp。在 Operand 中，并未有特殊的操作。

class Operand {
public:
    Operand();

    Operand(const Shape& shape, DataKind dataKind, const DeviceMesh& mesh, const PlacementSeq& placementSeq);

    ~Operand();

    DTORCH_DISABLE_COPY_AND_MOVE(Operand);

    DTORCH_FORCEINLINE void SetName(const std::string& name) { mName = name; }

    DTORCH_FORCEINLINE const std::string& GetName() const noexcept { return mName; }

    std::string GetTypeString() const;

    //-------------------------------------------- Topology -----------------------------------------------------------

    DTORCH_FORCEINLINE void SetProducerOp(Operator* op) { mProducerOp = op; }

    DTORCH_FORCEINLINE Operator* GetProducerOp() const noexcept { return mProducerOp; }

    DTORCH_FORCEINLINE const std::vector<Operator*>& GetConsumerOps() const noexcept { return mConsumerOps; }

    DTORCH_FORCEINLINE void AddConsumesOp(Operator* op) { mConsumerOps.push_back(op); }

    //-------------------------------------------- Meta Info -----------------------------------------------------------

    void SetShapeAndStride(const Shape& shape, const Stride& stride = Stride());

    DTORCH_FORCEINLINE const Shape& GetShape() const noexcept { return mShape; }

    DTORCH_FORCEINLINE bool IsNullTensorShape() const noexcept { return mShape.IsNullTensorShape(); }

    DTORCH_FORCEINLINE const Stride& GetStride() const noexcept { return mStride; }

    DTORCH_FORCEINLINE void SetDataKind(DataKind dataKind) { mDataKind = dataKind; }

    DTORCH_FORCEINLINE DataKind GetDataKind() const noexcept { return mDataKind; }

    void MetaDataSameAs(const Operand* otherPtr);

    //-------------------------------------------- Distribute ----------------------------------------------------------
    Shape GetLocalShape(int64_t globalDeviceId) const noexcept;

    bool IsLocalShapeEventSplit() const noexcept;

    DTORCH_FORCEINLINE bool IsDistributed() const { return mDeviceMesh.IsDistributed(); }

    void SetDeviceMeshAndPlacementSeq(const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq);

    DTORCH_FORCEINLINE DeviceKind GetDeviceKind() const noexcept { return mDeviceMesh.GetDeviceKind(); }

    DTORCH_FORCEINLINE const DeviceMesh& GetDeviceMesh() const noexcept { return mDeviceMesh; }

    DTORCH_FORCEINLINE const PlacementSeq& GetPlacementSeq() const noexcept { return mPlacementSeq; }

    const std::unordered_set<int64_t>& GetDistributedDeviceIdSet() const;

private:
    void ComputeLocalShape() const;

protected:
    std::string mName;

    // Topology
    Operator* mProducerOp;
    std::vector<Operator*> mConsumerOps;

    // Meta Info
    Shape mShape;
    Stride mStride;
    DataKind mDataKind;
    DeviceMesh mDeviceMesh;
    PlacementSeq mPlacementSeq;

    // Cached local shape
    mutable std::mutex mMutex;
    mutable std::unordered_map<int64_t, Shape> mAllLocalShape;
};

using OperandArray = std::vector<std::shared_ptr<Operand>>;

}  // namespace core
}  // namespace dtorch
