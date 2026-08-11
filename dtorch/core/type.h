/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "dtorch/api/cpp/api_type.h"
#include "dtorch/api/cpp/cluster.h"
#include "dtorch/api/cpp/data_kind.h"
#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/functional/functional_option.h"
#include "dtorch/api/cpp/generator.h"
#include "dtorch/api/cpp/graph.h"
#include "dtorch/api/cpp/index.h"
#include "dtorch/api/cpp/memory_stats.h"
#include "dtorch/api/cpp/params_hash.h"
#include "dtorch/api/cpp/scalar.h"
#include "dtorch/api/cpp/shape.h"
#include "dtorch/api/cpp/stride.h"

namespace dtorch {
namespace core {

using api::cpp::DataKind;
using api::cpp::DataKindFromString;
using api::cpp::DataKindSize;
using api::cpp::DataKindToString;
using api::cpp::DataKindTrait;
using api::cpp::DataTypeTrait;
using api::cpp::Device;
using api::cpp::DeviceKey;
using api::cpp::DeviceKeyMap;
using api::cpp::DeviceKeySet;
using api::cpp::DeviceKind;
using api::cpp::DeviceKindFromString;
using api::cpp::DeviceKindToString;
using api::cpp::DeviceMesh;
using api::cpp::DistributedSpec;
using api::cpp::Generator;
using api::cpp::GraphOption;
using api::cpp::Index;
using api::cpp::IntOrIntArray;
using api::cpp::MemoryStat;
using api::cpp::MemoryStats;
using api::cpp::OperatorFormat;
using api::cpp::PaddingType;
using api::cpp::ParamsEqual;
using api::cpp::ParamsHash;
using api::cpp::Partial;
using api::cpp::Placement;
using api::cpp::PlacementSeq;
using api::cpp::PoolingKind;
using api::cpp::Replicate;
using api::cpp::Scalar;
using api::cpp::Serialization;
using api::cpp::Shape;
using api::cpp::Shard;
using api::cpp::SimpleArray;
using api::cpp::Slice;
using api::cpp::Stride;
using api::cpp::distributed::NodeType;
using api::cpp::functional::SdpaOption;

// KernelStreamType
enum class KernelStreamType : int64_t {
    kCompute = 0,
    kCommunicate,
    kCount,
};

std::string KernelStreamTypeToString(KernelStreamType streamType);

std::ostream& operator<<(std::ostream& os, KernelStreamType streamType);

// KernelStreamKey
// KernelStreamKey MUST POD type
struct KernelStreamKey {
    DeviceKey device;
    KernelStreamType streamType;

    void Init(DeviceKind deviceKind, int64_t deviceId, KernelStreamType streamType) {
        // KernelStreamKey maybe align, set it all to 0
        std::memset(this, 0, sizeof(KernelStreamKey));

        this->device.deviceKind = deviceKind;
        this->device.deviceId = deviceId;
        this->streamType = streamType;
    }

    void Init(const Device& device, KernelStreamType streamType) {
        Init(device.deviceKind, device.deviceId, streamType);
    }

    void Init(const DeviceKey& device, KernelStreamType streamType) {
        Init(device.deviceKind, device.deviceId, streamType);
    }

    Device GetDevice() const noexcept { return Device(device.deviceKind, device.deviceId); }

    std::string ToString() const {
        std::stringstream ss;
        ss << "KernelStreamKey(" << device << ", " << streamType << ")";
        return ss.str();
    }

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const KernelStreamKey& streamKey) {
        os << streamKey.ToString();
        return os;
    }
};

static_assert(sizeof(KernelStreamKey) == sizeof(DeviceKey) + sizeof(KernelStreamType),
              "KernelStreamKey MUST POD type without memory align");

using StreamKeySet = std::unordered_set<KernelStreamKey, ParamsHash<KernelStreamKey>, ParamsEqual<KernelStreamKey>>;

template <class Type>
using StreamKeyMap =
    std::unordered_map<KernelStreamKey, Type, ParamsHash<KernelStreamKey>, ParamsEqual<KernelStreamKey>>;

}  // namespace core
}  // namespace dtorch
