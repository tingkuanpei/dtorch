/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "distributed_spec.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <torch/torch.h>

#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/simple_array.h"
#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/string.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace api {
namespace cpp {

DeviceMesh::DeviceMesh(DeviceKind deviceKind, const std::vector<int64_t>& meshVec,
                       const std::vector<std::string>& dimensionNames)
    : mDeviceKind(deviceKind), mMesh(std::make_shared<SimpleArray>(meshVec, dimensionNames)) {
    DDebugAssert(mMesh != nullptr);
}

DeviceMesh::DeviceMesh(const Device& device) : DeviceMesh(device.deviceKind, {device.deviceId}) {}

DeviceMesh::DeviceMesh(const torch::Device& torchDevice)
    : DeviceMesh(external::torch::TorchUtil::ToDevice(torchDevice)) {}

DeviceMesh::DeviceMesh(const torch::Device& device, const torch::Tensor& mesh,
                       const std::vector<std::string>& dimensionNames)
    : DeviceMesh(external::torch::TorchUtil::ToDevice(device).deviceKind, mesh, dimensionNames) {}

DeviceMesh::DeviceMesh(DeviceKind deviceKind, const torch::Tensor& mesh, const std::vector<std::string>& dimensionNames)
    : mDeviceKind(deviceKind), mMesh(std::make_shared<SimpleArray>(mesh, dimensionNames)) {
    DDebugAssert(mMesh != nullptr);
}

DeviceMesh::DeviceMesh(DeviceKind deviceKind, const SimpleArray& simpleArray)
    : mDeviceKind(deviceKind), mMesh(std::make_shared<SimpleArray>(simpleArray)) {
    DDebugAssert(mMesh != nullptr);
}

std::string DeviceMesh::ToString() const noexcept {
    DDebugAssert(mMesh != nullptr);
    std::stringstream ss;
    ss << "[DeviceKind: " << mDeviceKind << ", Mesh: " << *mMesh << "]";
    return ss.str();
}

bool DeviceMesh::operator==(const DeviceMesh& other) const noexcept {
    DDebugAssert(mMesh != nullptr);
    DDebugAssert(other.mMesh != nullptr);

    if (this == &other || (mDeviceKind == other.mDeviceKind && mMesh == other.mMesh)) {
        return true;
    }

    if (mDeviceKind != other.mDeviceKind || (*mMesh) != (*other.mMesh)) {
        return false;
    }

    return true;
}

bool DeviceMesh::IsContainDevice(const Device& device) const noexcept {
    if (device.deviceKind != GetDeviceKind() || !GetDeviceIdSet().count(device.deviceId)) {
        return false;
    }
    return true;
}

Device DeviceMesh::ToDevice() const noexcept {
    int64_t deviceId = 0;
    if (Count() == 1) {
        deviceId = mMesh->GetData()[0];
    }

    return Device(mDeviceKind, deviceId);
}

std::vector<Device> DeviceMesh::ToDeviceVec() const noexcept {
    std::vector<Device> result;
    for (auto id : mMesh->GetData()) {
        result.push_back(Device(mDeviceKind, id));
    }
    return result;
}

std::vector<DeviceMesh> DeviceMesh::Unbind(const std::vector<int64_t>& dims) const {
    auto simpleArrays = this->GetMesh().Unbind(dims);

    std::vector<DeviceMesh> result;
    for (size_t i = 0; i < simpleArrays.size(); i++) {
        result.push_back(DeviceMesh(this->GetDeviceKind(), simpleArrays[i]));
    }
    return result;
}

Placement::Placement(const std::string& placementStr)
    : mReplicate(false), mPartial(false), mShard(false), mShardIndex(-1), mSubSplitCoordinates(-1) {
    if (placementStr == "R") {
        mReplicate = true;
        return;
    } else if (placementStr == "P") {
        mPartial = true;
        return;
    } else if (placementStr == "S0") {
        mShard = true;
        mShardIndex = 0;
        return;
    } else if (placementStr == "S1") {
        mShard = true;
        mShardIndex = 1;
        return;
    } else if (placementStr == "S2") {
        mShard = true;
        mShardIndex = 2;
        return;
    } else if (placementStr.size() > 1 && placementStr[0] == 'S') {
        std::string shardIndex = placementStr.substr(1);
        try {
            mShard = true;
            mShardIndex = std::stoi(shardIndex);
            return;
        } catch (std::exception& e) {
        }
    }

    mReplicate = true;
    DLogFatal() << "Unsupport placemanet: " << placementStr;
    DUnimplemented();
}

Placement::Placement(bool replicate, bool partial, bool shard, int64_t shardIndex, int64_t subSplitCoordinates)
    : mReplicate(replicate),
      mPartial(partial),
      mShard(shard),
      mShardIndex(shardIndex),
      mSubSplitCoordinates(subSplitCoordinates) {}

Placement Shard(int64_t shardIndex, int64_t subSplitCoordinates) {
    return Placement(false, false, true, shardIndex, subSplitCoordinates);
}

Placement Replicate() { return Placement(true, false, false, -1, -1); }

Placement Partial() { return Placement(false, true, false, -1, -1); }

std::string Placement::ToString() const noexcept {
    std::stringstream ss;
    if (mReplicate) {
        ss << "Replicate()";
    } else if (mPartial) {
        ss << "Partial(sum)";
    } else if (mShard) {
        if (mSubSplitCoordinates == -1) {
            ss << "Shard(dim=" << mShardIndex << ")";
        } else {
            ss << "Shard(dim=" << mShardIndex << "," << " sub_split_coordinates=" << mSubSplitCoordinates << ")";
        }
    } else {
        ss << "None";
    }
    return ss.str();
}

uint64_t Placement::ToHashKey() const noexcept {
    uint64_t result = 0;
    int64_t shardIndex = mShard ? mShardIndex : 0;
    DAlwaysAssertMsg(shardIndex >= 0 && shardIndex <= 0x0F, "ShardIndex too large: " + std::to_string(shardIndex));
    result = mReplicate << 0 | mPartial << 1 | mShard << 2 | ((mShardIndex & 0x0F) << 4);
    return result;

    // TODO: support mSubSplitCoordinates
    // DAlwaysAssert(mSubSplitCoordinates >= -1);
    // bool isSubSplitCoordinates = mSubSplitCoordinates != -1;
    // int64_t subSplitCoordinates = isSubSplitCoordinates ? mSubSplitCoordinates : 0;
    // DAlwaysAssertMsg(subSplitCoordinates >= 0 && subSplitCoordinates <= 0xFFFFFFFF,
    //                  "SubSplitCoordinates too large: " + std::to_string(mSubSplitCoordinates));
    // result = mReplicate << 0 | mPartial << 1 | mShard << 2 | isSubSplitCoordinates << 3 | ((mShardIndex & 0x0F) << 4)
    // |
    //          ((mSubSplitCoordinates & 0xFFFFFFFF) << 8);
    // return result;
}

bool Placement::operator==(const Placement& other) const noexcept {
    return mReplicate == other.mReplicate && mPartial == other.mPartial && mShard == other.mShard &&
           mShardIndex == other.mShardIndex && mSubSplitCoordinates == other.mSubSplitCoordinates;
}

std::vector<int64_t> Placement::GetShardSizeForAllRank(int64_t shapeSize, int64_t numShard) {
    DDebugAssert(shapeSize >= numShard);
    std::vector<int64_t> result(numShard, shapeSize / numShard);
    if (shapeSize % numShard == 0) {
        return result;
    } else {
        for (int64_t i = 0; i < shapeSize % numShard; i++) {
            result[i]++;
        }
        return result;
    }
}

PlacementSeq::PlacementSeq() : mData() { mData = std::make_shared<std::vector<Placement>>(); }

PlacementSeq::PlacementSeq(const std::vector<std::string>& placementSeqStr) : mData() {
    std::vector<Placement> placementVec;
    for (const auto& it : placementSeqStr) {
        placementVec.push_back(Placement(it));
    }
    mData = std::make_shared<std::vector<Placement>>(placementVec);
}

bool PlacementSeq::IsAllReplicate() const noexcept {
    DDebugAssert(mData != nullptr);
    if (mData->size() == 0) {
        return false;
    }
    for (const auto& it : *mData) {
        if (!it.IsReplicate()) {
            return false;
        }
    }
    return true;
}

bool PlacementSeq::HasPartial() const noexcept {
    DDebugAssert(mData != nullptr);
    if (mData->size() == 0) {
        return false;
    }
    for (const auto& it : *mData) {
        if (it.IsPartial()) {
            return true;
        }
    }
    return false;
}

bool PlacementSeq::operator==(const PlacementSeq& other) const noexcept {
    DDebugAssert(mData != nullptr);
    DDebugAssert(other.mData != nullptr);
    if (Size() != other.Size()) {
        return false;
    }

    for (size_t i = 0; i < Size(); i++) {
        if ((*mData)[i] != (*other.mData)[i]) {
            return false;
        }
    }
    return true;
}

Placement& PlacementSeq::operator[](int64_t index) {
    DDebugAssert(mData != nullptr);
    mData = std::make_shared<std::vector<Placement>>(mData->begin(), mData->end());
    return (*mData)[index];
}

std::vector<Placement> PlacementSeq::Vec() const noexcept {
    DDebugAssert(mData != nullptr);
    std::vector<Placement> result;
    for (const auto& it : *mData) {
        result.push_back(it);
    }
    return result;
}

std::string PlacementSeq::ToString() const noexcept {
    DDebugAssert(mData != nullptr);
    std::stringstream ss;
    ss << "[";
    for (const auto& it : *mData) {
        ss << it;
        ss << ", ";
    }
    ss << "]";
    return ss.str();
}

void PlacementSeq::ToReplicateWhenDimSizeEqualOne(const DeviceMesh& deviceMesh) {
    DDebugAssert(mData != nullptr);
    DDebugAssert(mData->size() == deviceMesh.NumAxis());

    const Shape& meshShape = deviceMesh.GetShape();
    for (size_t i = 0; i < deviceMesh.NumAxis(); i++) {
        if (meshShape[i] == 1) {
            (*mData)[i] = Replicate();
        }
    }
}

void PlacementSeq::ShardDimToPositiveNumber(const Shape& shape) {
    DDebugAssert(mData != nullptr);

    for (auto& it : (*mData)) {
        if (it.IsShard() && it.GetShardIndex() < 0 &&
            it.GetShardIndex() >= (-1 * static_cast<int64_t>(shape.NumAxis()))) {
            it = Shard(it.GetShardIndex() + shape.NumAxis());
        }
    }
}

std::vector<size_t> PlacementSeq::GetDiffDims(const PlacementSeq& other) const noexcept {
    DDebugAssert(Size() == other.Size());
    std::vector<size_t> result;
    for (size_t i = 0; i < Size(); i++) {
        if ((*this)[i] != other[i]) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<size_t> PlacementSeq::GetConvertOrder(const PlacementSeq& destPlacements) {
    std::vector<size_t> b2f;
    std::vector<size_t> result;

    for (size_t i = 0; i < Size(); i++) {
        const auto& placementA = (*mData)[i];
        const auto& placementB = destPlacements[i];
        if (placementA.IsShard() && placementB.IsReplicate()) {
            b2f.push_back(i);
        } else if (placementA != placementB) {
            result.push_back(i);
        }
    }

    result.insert(result.end(), b2f.rbegin(), b2f.rend());
    return result;
}

bool DistributedSpec::CheckValid(const Shape& shape, const DeviceMesh& deviceMesh, const PlacementSeq& placementSeq,
                                 int64_t totalGpuCount) {
    if (shape.IsNullTensorShape()) {
        return true;
    }

    if (placementSeq.Size() <= 0) {
        DLogError() << "size of placementSeq is zero";
        return false;
    }

    if (!CheckDeviceIdValid(deviceMesh, totalGpuCount)) {
        return false;
    }

    if (deviceMesh.NumAxis() != placementSeq.Size()) {
        DLogError() << "DeviceMesh and PlacementSeq have different size: " << deviceMesh.NumAxis() << " vs "
                    << placementSeq.Size();
        return false;
    }

    if (deviceMesh.Count() == 1 && !placementSeq[0].IsReplicate()) {
        DLogError() << "DeviceMesh only have one device id, but placementSeq isn't replicate";
        return false;
    }

    const std::unordered_set<int64_t>& meshSet = deviceMesh.GetDeviceIdSet();
    if (meshSet.size() != deviceMesh.GetMesh().Count()) {
        DLogError() << "DeviceMesh has duplicate device id";
        return false;
    }

    for (size_t i = 0; i < deviceMesh.NumAxis(); i++) {
        if (placementSeq[i].IsShard()) {
            int64_t shardIdx = placementSeq[i].GetShardIndex();
            int64_t subSplitCoordinates = placementSeq[i].GetSubSplitCoordinates();
            if (shardIdx < 0) {
                DLogError() << "shard dim less than zero: " << shardIdx;
                return false;
            }
            if (shardIdx >= static_cast<int64_t>(shape.NumAxis())) {
                DLogError() << "shard dim larger than shape dimention: " << shardIdx;
                return false;
            }
            int64_t shardSize = deviceMesh.GetMesh().GetShape()[i];
            if (shardSize < 2) {
                DLogError() << "shard size less than 2";
                return false;
            }

            if (subSplitCoordinates != -1 && subSplitCoordinates < 0) {
                DLogError() << "subSplitCoordinates should = -1 or larger than zero: " << subSplitCoordinates;
                return false;
            }
        }
    }

    if (!CheckShapeValid(shape, deviceMesh, placementSeq)) {
        return false;
    }

    return true;
}

bool DistributedSpec::CheckShapeValid(const Shape& shape, const DeviceMesh& deviceMesh,
                                      const PlacementSeq& placementSeq) {
    // TODO: support Check SubSplitCoordinates
    // TODO: remove GetShardInfo
    auto shardInfoVec = DistributedSpec::GetShardInfo(deviceMesh, placementSeq);
    for (const auto& info : shardInfoVec) {
        size_t shardDim = info.shardDim;
        if (shape[shardDim] < static_cast<int64_t>(info.shardSize)) {
            DLogError() << "Can't shard tensor with index: " << shardDim << ", size to shard: " << shape[shardDim]
                        << ", num of shard: " << info.shardSize << ", shape=" << shape
                        << ", placements=" << placementSeq;
            return false;
        }
    }
    return true;
}

bool DistributedSpec::CheckDeviceIdValid(const DeviceMesh& deviceMesh, int64_t totalGpuCount, bool logErrorStr) {
    if (deviceMesh.GetDeviceKind() == DeviceKind::kCpu) {
        return true;
    }

    for (auto id : deviceMesh.GetDeviceIdSet()) {
        if (id >= totalGpuCount) {
            if (logErrorStr) {
                DLogError() << "Device id(" << id << ") MUST small than gpu device count(" << totalGpuCount << ")";
            }
            return false;
        }
    }
    return true;
}

void DistributedSpec::MergeSameShardDim(DeviceMesh& deviceMesh, PlacementSeq& placementSeq) {
    auto shardInfoVec = DistributedSpec::GetShardInfo(deviceMesh, placementSeq);
    std::unordered_map<size_t, std::vector<size_t>> needMergeDimMap;
    for (const auto& info : shardInfoVec) {
        if (info.meshDims.size() > 1) {
            std::vector<size_t> meshDimsVec(info.meshDims.begin(), info.meshDims.end());
            std::sort(meshDimsVec.begin(), meshDimsVec.end());
            needMergeDimMap[meshDimsVec[0]] = meshDimsVec;
        }
    }

    if (needMergeDimMap.size() == 0) {
        return;
    }

    Shape meshShape = deviceMesh.GetMeshShape();
    Shape newMeshShape;
    std::unordered_set<size_t> usedDim;
    std::vector<int64_t> permuteDim;
    std::vector<Placement> newPlacements;

    for (size_t i = 0; i < meshShape.NumAxis(); i++) {
        if (usedDim.find(i) == usedDim.end()) {
            if (needMergeDimMap.find(i) == needMergeDimMap.end()) {
                permuteDim.push_back(i);
                usedDim.insert(i);
                newMeshShape.PushBack(meshShape[i]);
                newPlacements.push_back(placementSeq[i]);
            } else {
                size_t size = 1;
                for (auto it : needMergeDimMap.at(i)) {
                    DDebugAssert(usedDim.find(it) == usedDim.end());
                    permuteDim.push_back(it);
                    size *= meshShape[it];
                    usedDim.insert(it);
                }
                newMeshShape.PushBack(size);
                newPlacements.push_back(placementSeq[i]);
            }
        }
    }
    DDebugAssert(permuteDim.size() == meshShape.NumAxis());

    torch::Tensor meshTensor = deviceMesh.GetMesh().ToTrochTensor();
    meshTensor = meshTensor.permute(permuteDim).reshape(newMeshShape.Vec());
    deviceMesh = DeviceMesh(deviceMesh.GetDeviceKind(), meshTensor);
    placementSeq = PlacementSeq(newPlacements);
}

std::vector<DistributedSpec::ShardInfo> DistributedSpec::GetShardInfo(const DeviceMesh& deviceMesh,
                                                                      const PlacementSeq& placementSeq) {
    std::unordered_map<size_t, ShardInfo> resultMap;
    Shape meshShape = deviceMesh.GetShape();
    DDebugAssert(deviceMesh.NumAxis() == placementSeq.Size());

    for (size_t i = 0; i < deviceMesh.NumAxis(); i++) {
        if (placementSeq[i].IsShard()) {
            int64_t shardIndex = placementSeq[i].GetShardIndex();
            DDebugAssert(shardIndex >= 0);

            if (resultMap.find(shardIndex) == resultMap.end()) {
                ShardInfo shardInfo;
                shardInfo.shardDim = static_cast<size_t>(shardIndex);
                shardInfo.shardSize = meshShape[i];
                shardInfo.meshDims.insert(i);
                resultMap[shardIndex] = shardInfo;
            } else {
                ShardInfo& shardInfo = resultMap.at(shardIndex);
                DDebugAssert(shardInfo.shardDim == static_cast<size_t>(shardIndex));
                shardInfo.shardSize *= meshShape[i];
                shardInfo.meshDims.insert(i);
            }
        }
    }

    std::vector<ShardInfo> result;
    for (const auto& it : resultMap) {
        result.push_back(it.second);
    }
    return result;
}

size_t DistributedSpec::IndexInSelectedDims(size_t index, const Stride& stride,
                                            const std::unordered_set<size_t>& selectedDims,
                                            const Stride& selectedDimStride) {
    DDebugAssert(selectedDimStride.NumAxis() == selectedDims.size());

    auto coordinate = stride.ComputeCoordinate(index);
    std::vector<size_t> selectCoordinate;
    for (size_t i = 0; i < coordinate.size(); i++) {
        if (selectedDims.find(i) != selectedDims.end()) {
            selectCoordinate.push_back(coordinate[i]);
        }
    }
    return selectedDimStride.ComputeIndex(selectCoordinate);
}

std::unordered_map<int64_t, Shape> DistributedSpec::ComputeLocalShape(const Shape& shape, const DeviceMesh& deviceMesh,
                                                                      const PlacementSeq& placementSeq) {
    std::unordered_map<int64_t, Shape> result;
    bool evenSplit = true;
    Shape evenSplitShape = shape;
    Shape meshShape = deviceMesh.GetShape();
    auto shardInfoVec = DistributedSpec::GetShardInfo(deviceMesh, placementSeq);

    for (const auto& info : shardInfoVec) {
        int64_t shardDim = info.shardDim;
        if (shape[shardDim] % info.shardSize != 0) {
            evenSplit = false;
        } else {
            evenSplitShape[shardDim] = shape[shardDim] / info.shardSize;
        }
    }

    // Shape is evenSplited, return evenSplitShape directly.
    if (evenSplit) {
        result[0] = evenSplitShape;
        return result;
    }

    // Shape isn't evenSplited
    const std::vector<int64_t>& meshData = deviceMesh.GetMesh().GetData();
    for (auto it : meshData) {
        result[it] = evenSplitShape;
    }

    Stride meshStride(meshShape);
    for (const auto& info : shardInfoVec) {
        int64_t shardDim = info.shardDim;
        if (shape[shardDim] % info.shardSize == 0) {
            continue;
        }

        Stride selectedStride = Stride(meshShape.KeepSelectedDim(info.meshDims));
        std::vector<int64_t> splitSize = Placement::GetShardSizeForAllRank(shape[shardDim], info.shardSize);
        DDebugAssert(splitSize.size() == info.shardSize);

        for (size_t dataIdx = 0; dataIdx < meshData.size(); dataIdx++) {
            size_t indexInSelectedDims = IndexInSelectedDims(dataIdx, meshStride, info.meshDims, selectedStride);
            DDebugAssert(indexInSelectedDims < splitSize.size());
            result[meshData[dataIdx]][shardDim] = splitSize[indexInSelectedDims];
        }
    }

    return result;
}

size_t DistributedSpec::GetRankId(int64_t currentId, const std::vector<int64_t>& allDeviceId) {
    for (size_t i = 0; i < allDeviceId.size(); i++) {
        if (currentId == allDeviceId[i]) {
            return i;
        }
    }
    DLogFatal() << "Can't find id(" << currentId << ") in device list: " << String::ToString(allDeviceId);
    DUnsupportedImpl();
    return 0;
}

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
