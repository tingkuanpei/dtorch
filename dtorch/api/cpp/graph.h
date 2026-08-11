/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "dtorch/api/cpp/data_kind.h"
#include "dtorch/api/cpp/device.h"
#include "dtorch/api/cpp/distributed_spec.h"
#include "dtorch/api/cpp/void_future_collect.h"
#include "serialization.h"

namespace dtorch {
namespace core {
class GraphConstructor;
namespace distributed {
class ClusterInfo;
}  // namespace distributed
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace api {
namespace cpp {

struct GraphOption {
    std::optional<bool> perDevicePerProcess;

public:
    GraphOption() : perDevicePerProcess(std::nullopt), graphId(std::nullopt) {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & perDevicePerProcess;
        ar & graphId;
    }

    std::string ToString() const;

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const GraphOption& graphOption) {
        os << graphOption.ToString();
        return os;
    }

    // Boost-text-archive (de)serialization for cross-node transfer via gRPC (CreateGraph).
    std::string SerializeToString() const;

    bool operator==(const GraphOption& other) const;

    DTORCH_API_FORCEINLINE bool operator!=(const GraphOption& other) const { return !(*this == other); }

public:
    static GraphOption UpdateOptionFromEnvironment(const GraphOption& referenceOption);

    static GraphOption FromString(const std::string& serializedData);

    // Only use internally
    std::optional<uint64_t> graphId;
};

class Graph {
public:
    Graph(const GraphOption& graphOption = GraphOption());

    ~Graph();

    void Destroy();

    uint64_t GetId() const noexcept;

    void SetName(const std::string& graphName);

    const std::string& GetName();

    // Async sync: returns VoidFutureCollect immediately. Call Wait() to block.
    VoidFutureCollect SyncFuture();

    // Blocking sync: waits until all devices are synchronized.
    // Convenience wrapper around SyncFuture().Wait(), preserves old API.
    void Sync();

    void SetDefaultDeviceMesh(const DeviceMesh& deviceMesh);

    const DeviceMesh& GetDefaultDeviceMesh() const noexcept;

    bool Satisfy(const DeviceMesh& deviceMesh) const noexcept;

    void SetDefaultDataKind(DataKind dataKind);

    const DataKind& GetDefaultDataKind() const noexcept;

    const GraphOption& GetGraphOption() const noexcept;

public:
    static Graph GetDefaultThreadLocalGraph();

public:
    // Function only call from internal
    core::GraphConstructor* GetGraphConstructor() const noexcept;

    const core::distributed::ClusterInfo& GetClusterInfo() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> mImplPtr;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
