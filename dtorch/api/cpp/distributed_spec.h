/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "api_type.h"
#include "api_utilities.h"
#include "device.h"
#include "shape.h"
#include "simple_array.h"
#include "stride.h"

namespace dtorch {
namespace api {
namespace cpp {

class DeviceMesh {
public:
    explicit DeviceMesh(const Device& device = Device::GetDefaultCpuDevice());

    explicit DeviceMesh(const torch::Device& torchDevice);

    DeviceMesh(DeviceKind deviceKind, const std::vector<int64_t>& meshVec,
               const std::vector<std::string>& dimensionNames = {});

    DeviceMesh(const torch::Device& device, const torch::Tensor& mesh,
               const std::vector<std::string>& dimensionNames = {});

    DeviceMesh(DeviceKind deviceKind, const torch::Tensor& mesh, const std::vector<std::string>& dimensionNames = {});

    DeviceMesh(DeviceKind deviceKind, const SimpleArray& simpleArray);

    DTORCH_API_FORCEINLINE DeviceKind GetDeviceKind() const noexcept { return mDeviceKind; }

    DTORCH_API_FORCEINLINE const SimpleArray& GetMesh() const noexcept { return *mMesh; }

    DTORCH_API_FORCEINLINE const Shape& GetMeshShape() const noexcept { return GetMesh().GetShape(); }

    DTORCH_API_FORCEINLINE const std::unordered_set<int64_t>& GetDeviceIdSet() const noexcept {
        return mMesh->GetDataSet();
    }

    std::string ToString() const noexcept;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const DeviceMesh& mesh) {
        os << mesh.ToString();
        return os;
    }

    bool operator==(const DeviceMesh& other) const noexcept;

    DTORCH_API_FORCEINLINE bool operator!=(const DeviceMesh& other) const noexcept {
        return !(this->operator==(other));
    }

    DTORCH_API_FORCEINLINE size_t NumAxis() const noexcept { return mMesh->NumAxis(); }

    DTORCH_API_FORCEINLINE size_t Count() const noexcept { return mMesh->Count(); }

    DTORCH_API_FORCEINLINE bool IsDistributed() const { return Count() > 1; }

    DTORCH_API_FORCEINLINE const Shape& GetShape() const noexcept { return mMesh->GetShape(); }

    bool IsContainDevice(const Device& device) const noexcept;

    Device ToDevice() const noexcept;

    std::vector<Device> ToDeviceVec() const noexcept;

    std::vector<DeviceMesh> Unbind(const std::vector<int64_t>& dims) const;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mDeviceKind;
        ar & mMesh;
    }

private:
    DeviceKind mDeviceKind;
    // Using std::shared_ptr to prevent the overhead of deep copy.
    std::shared_ptr<const SimpleArray> mMesh;
};

class Placement {
public:
    Placement(const std::string& placementStr = "R");

    DTORCH_API_FORCEINLINE bool IsShard() const noexcept { return mShard; }

    DTORCH_API_FORCEINLINE bool IsShard(int64_t shardIndex) const noexcept {
        return mShard && mShardIndex == shardIndex;
    }

    DTORCH_API_FORCEINLINE int64_t GetShardIndex() const noexcept {
        DApiAssert(mShard);
        return mShardIndex;
    }

    DTORCH_API_FORCEINLINE bool HasSubSplitCoordinates() const noexcept {
        return IsShard() && GetSubSplitCoordinates() != -1;
    }

    DTORCH_API_FORCEINLINE int64_t GetSubSplitCoordinates() const noexcept {
        DApiAssert(mShard);
        return mSubSplitCoordinates;
    }

    DTORCH_API_FORCEINLINE void SetSubSplitCoordinates(int64_t value) noexcept {
        DApiAssert(mShard);
        mSubSplitCoordinates = value;
    }

    DTORCH_API_FORCEINLINE bool IsReplicate() const noexcept { return mReplicate; }

    DTORCH_API_FORCEINLINE bool IsPartial() const noexcept { return mPartial; }

    bool operator==(const Placement& other) const noexcept;

    DTORCH_API_FORCEINLINE bool operator!=(const Placement& other) const noexcept { return !(this->operator==(other)); }

    std::string ToString() const noexcept;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const Placement& placement) {
        os << placement.ToString();
        return os;
    }

    uint64_t ToHashKey() const noexcept;

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mReplicate;
        ar & mPartial;
        ar & mShard;
        ar & mShardIndex;
        ar & mSubSplitCoordinates;
    }

private:
    Placement(bool replicate, bool partial, bool shard, int64_t shardIndex, int64_t subSplitCoordinates);

    friend Placement Shard(int64_t shardIndex, int64_t subSplitCoordinates);
    friend Placement Replicate();
    friend Placement Partial();

public:
    // Function in this part only call from internal
    static Placement Optional() { return Placement(false, false, false, -1, -1); }

    static std::vector<int64_t> GetShardSizeForAllRank(int64_t shapeSize, int64_t numShard);

private:
    bool mReplicate;
    bool mPartial;
    bool mShard;
    int64_t mShardIndex;
    int64_t mSubSplitCoordinates;
};

Placement Shard(int64_t shardIndex, int64_t subSplitCoordinates = -1);

Placement Replicate();

Placement Partial();

class PlacementSeq {
public:
    PlacementSeq();

    PlacementSeq(const std::vector<std::string>& placementSeqStr);

    PlacementSeq(const std::initializer_list<Placement>& args)
        : mData(std::make_shared<std::vector<Placement>>(args)) {}

    PlacementSeq(const std::vector<Placement>& args) : mData(std::make_shared<std::vector<Placement>>(args)) {}

    PlacementSeq(size_t count, const Placement& value)
        : mData(std::make_shared<std::vector<Placement>>(count, value)) {}

    bool IsAllReplicate() const noexcept;

    bool HasPartial() const noexcept;

    DTORCH_API_FORCEINLINE size_t Size() const noexcept { return mData->size(); }

    bool operator==(const PlacementSeq& other) const noexcept;

    DTORCH_API_FORCEINLINE bool operator!=(const PlacementSeq& other) const noexcept {
        return !(this->operator==(other));
    }

    const Placement& operator[](int64_t index) const noexcept { return (*mData)[index]; }

    Placement& operator[](int64_t index);

    std::vector<Placement> Vec() const noexcept;

    std::string ToString() const noexcept;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const PlacementSeq& placementSeq) {
        os << placementSeq.ToString();
        return os;
    }

    void ToReplicateWhenDimSizeEqualOne(const DeviceMesh& deviceMesh);

    void ShardDimToPositiveNumber(const Shape& shape);

    std::vector<size_t> GetDiffDims(const PlacementSeq& other) const noexcept;

    // Shard -> Replicate in same dimension need gather from back to front
    // Replicate -> Shard in same dimension need scather from front to back
    std::vector<size_t> GetConvertOrder(const PlacementSeq& destPlacements);

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mData;
    }

private:
    // Using std::shared_ptr to prevent the overhead of deep copy.
    std::shared_ptr<std::vector<Placement>> mData;
};

struct DistributedSpec {
public:
    static bool CheckValid(const Shape& shape, const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq,
                           int64_t totalGpuCount);

    static bool CheckShapeValid(const Shape& shape, const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq);

    static bool CheckDeviceIdValid(const DeviceMesh& deviceMesh, int64_t totalGpuCount, bool logErrorStr = true);

    // Merge placement shard in same dimention
    // input: device_mesh = [[0, 1], [2, 3]], placement = [S1, S1]
    // output: device_mesh = [0, 1, 2, 3], placement = [S1]
    static void MergeSameShardDim(DeviceMesh& deviceMesh, PlacementSeq& placementSeq);

    struct ShardInfo {
        size_t shardDim;
        size_t shardSize;
        std::unordered_set<size_t> meshDims;

    public:
        ShardInfo() : shardDim(0), shardSize(0), meshDims() {}
    };

    static std::vector<ShardInfo> GetShardInfo(const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq);

    static size_t IndexInSelectedDims(size_t index, const Stride& stride, const std::unordered_set<size_t>& selectDims,
                                      const Stride& selectDimStride);

    static std::unordered_map<int64_t, Shape> ComputeLocalShape(const Shape& shape, const DeviceMesh& deviceMesh,
                                                                const PlacementSeq& placementSeq);

    static size_t GetRankId(int64_t currentId, const std::vector<int64_t>& allDeviceId);
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
