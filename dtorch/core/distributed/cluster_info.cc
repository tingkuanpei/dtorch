/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "cluster_info.h"

#include <sstream>

#include "dtorch/api/cpp/global_option.h"
#include "dtorch/common/debug.h"
#include "dtorch/external/boost/boost_serialization.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {
namespace distributed {

NodeInfo NodeInfo::GetMainNodeInfo() {
    NodeInfo nodeInfo;
    nodeInfo.nodeType = NodeType::kMain;
    nodeInfo.rank = 0;
    nodeInfo.gpuCount = static_cast<int64_t>(external::torch::TorchUtil::CudaDeviceCount());
    return nodeInfo;
}

NodeInfo NodeInfo::GetWorkerNodeInfo(int64_t gpuCount) {
    NodeInfo nodeInfo;
    nodeInfo.nodeType = NodeType::kWorker;
    nodeInfo.gpuCount = gpuCount;
    return nodeInfo;
}

std::string NodeInfo::ToString() const {
    std::stringstream ss;
    ss << "NodeInfo(nodeType: " << nodeType << ", rank: " << rank << ", gpuCount: " << gpuCount << ")";
    return ss.str();
}

ClusterInfo::ClusterInfo(const std::string& mainNodeRpcAddress)
    : mMainNodeRpcAddress(mainNodeRpcAddress), mTotalGpuCount(0), mNodeInfoVec(), mDTensorInSameDevice(false) {
    PushBack(NodeInfo::GetMainNodeInfo());

    if (core::GlobalOption::GetSingleton().GetDTensorInSameDevice()) {
        mDTensorInSameDevice = true;
        if (mTotalGpuCount > 0) {
            int64_t totalGpuCount = core::GlobalOption::GetSingleton().GetNumGpuWhenEnableDtensorInSameDevice();
            DDebugAssert(totalGpuCount > 0);
            mNodeInfoVec[0].gpuCount = totalGpuCount;
            mTotalGpuCount = totalGpuCount;
        }
    }
}

std::vector<RunnerSupportedDevices> ClusterInfo::GetRunnerSupportedDevicesForNode() const noexcept {
    std::vector<RunnerSupportedDevices> result;

    int64_t counter = 0;
    for (size_t i = 0; i < NodeSize(); i++) {
        std::vector<DevicePair> devicePairForOneNode;

        for (int64_t gpuIdx = 0; gpuIdx < mNodeInfoVec[i].gpuCount; gpuIdx++) {
            DevicePair pair;
            pair.globalDevice = Device(DeviceKind::kGpu, counter);
            if (mDTensorInSameDevice) {
                pair.localDevice = Device(DeviceKind::kGpu, 0);
            } else {
                pair.localDevice = Device(DeviceKind::kGpu, gpuIdx);
            }

            counter++;
            devicePairForOneNode.push_back(pair);
        }

        // Only first node support cpu
        bool supportCpu = (i == 0);
        result.push_back(RunnerSupportedDevices(devicePairForOneNode, supportCpu));
    }

    return result;
}

std::string ClusterInfo::ToString() const {
    std::stringstream ss;
    ss << "ClusterInfo(mainNodeRpcAddress: " << mMainNodeRpcAddress << ", totalGpuCount: " << mTotalGpuCount
       << ", nodeInfoVec: [";
    for (size_t i = 0; i < mNodeInfoVec.size(); i++) {
        ss << mNodeInfoVec[i];
        if (i != mNodeInfoVec.size() - 1) {
            ss << ", ";
        }
    }
    ss << "], dtensorInSameDevice: " << mDTensorInSameDevice << ")";
    return ss.str();
}

std::string ClusterInfo::SerializeToString() const {
    std::ostringstream oss;
    boost::archive::text_oarchive oa(oss);
    oa << *this;
    return oss.str();
}

ClusterInfo ClusterInfo::FromString(const std::string& serializedData) {
    std::istringstream iss(serializedData);
    boost::archive::text_iarchive ia(iss);
    ClusterInfo clusterInfo;
    ia >> clusterInfo;
    return clusterInfo;
}

void ClusterInfo::InitFromString(const std::string& serializedData) { GetSingleton() = FromString(serializedData); }

}  // namespace distributed
}  // namespace core
}  // namespace dtorch
