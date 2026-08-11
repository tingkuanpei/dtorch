/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/runner/runner_supported_devices.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {
namespace distributed {

class NodeInfo {
public:
    static NodeInfo GetMainNodeInfo();

    static NodeInfo GetWorkerNodeInfo(int64_t gpuCount);

public:
    NodeInfo() : nodeType(NodeType::kMain), rank(-1), gpuCount(0) {}

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const NodeInfo& nodeInfo) {
        os << nodeInfo.ToString();
        return os;
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        // nodeType is an enum class — serialize it via the underlying integer type.
        int64_t nt = static_cast<int64_t>(nodeType);
        ar & nt;
        nodeType = static_cast<NodeType>(nt);
        ar & rank;
        ar & gpuCount;
    }

public:
    NodeType nodeType;
    int64_t rank;
    int64_t gpuCount;
};

class ClusterInfo {
public:
    // Global singleton. On the MainNode it is populated during cluster formation (see MainNode);
    // on WorkerNode processes and RemoteRunner subprocesses it is populated from the serialized
    // form shipped via CreateGraph / the subprocess launch payload.
    DTORCH_FORCEINLINE static ClusterInfo& GetSingleton() {
        static ClusterInfo clusterInfo;
        return clusterInfo;
    }

    // Deserialize into a fresh value (e.g. gRPC server side, before assigning into the singleton).
    static ClusterInfo FromString(const std::string& serializedData);

    // Populate the singleton from a serialized string (WorkerNode / subprocess side).
    static void InitFromString(const std::string& serializedData);

public:
    ClusterInfo() : mMainNodeRpcAddress(""), mTotalGpuCount(0), mNodeInfoVec(), mDTensorInSameDevice(false) {}

    ClusterInfo(const std::string& mainNodeRpcAddress);

    std::vector<RunnerSupportedDevices> GetRunnerSupportedDevicesForNode() const noexcept;

    DTORCH_FORCEINLINE const std::string& GetMainNodeRpcAddress() const noexcept { return mMainNodeRpcAddress; }

    DTORCH_FORCEINLINE size_t NodeSize() const noexcept { return mNodeInfoVec.size(); }

    DTORCH_FORCEINLINE std::vector<Device> GetAllDevices() const noexcept {
        std::vector<Device> devices;
        devices.push_back(Device(DeviceKind::kCpu, 0));
        auto gpuDevices = GetGpuDevices();
        devices.insert(devices.end(), gpuDevices.begin(), gpuDevices.end());
        return devices;
    }

    DTORCH_FORCEINLINE std::vector<Device> GetGpuDevices() const noexcept {
        std::vector<Device> devices;
        for (int64_t i = 0; i < GetTotalGpuCount(); i++) {
            devices.push_back(Device(DeviceKind::kGpu, i));
        }
        return devices;
    }

    DTORCH_FORCEINLINE int64_t GetTotalGpuCount() const noexcept { return mTotalGpuCount; }

    DTORCH_FORCEINLINE bool GetDTensorInSameDevice() const noexcept { return mDTensorInSameDevice; }

    DTORCH_FORCEINLINE const NodeInfo& Node(size_t idx) const noexcept {
        DDebugAssert(idx < mNodeInfoVec.size());
        return mNodeInfoVec[idx];
    }

    DTORCH_FORCEINLINE void PushBack(const NodeInfo& node) {
        mNodeInfoVec.push_back(node);
        DDebugAssert(node.gpuCount >= 0);
        mTotalGpuCount += node.gpuCount;
    }

    // Serialize to string for cross-process transfer (text archive, see GlobalOption).
    std::string SerializeToString() const;

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const ClusterInfo& clusterInfo) {
        os << clusterInfo.ToString();
        return os;
    }

    friend Serialization;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mMainNodeRpcAddress;
        ar & mTotalGpuCount;
        ar & mNodeInfoVec;
        ar & mDTensorInSameDevice;
    }

private:
    std::string mMainNodeRpcAddress;
    int64_t mTotalGpuCount;
    std::vector<NodeInfo> mNodeInfoVec;
    bool mDTensorInSameDevice;
};

}  // namespace distributed
}  // namespace core
}  // namespace dtorch
