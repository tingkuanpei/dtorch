/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "api_utilities.h"

namespace dtorch {
namespace core {
namespace distributed {
class MainNode;
class WorkerNode;
}  // namespace distributed
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace api {
namespace cpp {
namespace distributed {

enum class NodeType {
    kMain = 0,
    kWorker,
    kCount,
};

std::string NodeTypeToString(NodeType nodeType);

std::ostream& operator<<(std::ostream& os, NodeType nodeType);

class MainNode;
class WorkerNode;

class Cluster {
public:
    DTORCH_API_FORCEINLINE static Cluster& GetSingleton() {
        static Cluster cluster;
        return cluster;
    }

    static std::string GetValidNodeAddress();

public:
    bool IsCreate() const noexcept;

    NodeType GetNodeType() const noexcept;

    const std::string& GetMainNodeAddress() const noexcept;

    const std::string& GetMainProcessHeartBeatAddress() const noexcept;

private:
    struct Impl;
    mutable std::shared_ptr<Impl> mImplPtr;

private:
    Cluster();

    ~Cluster();

    friend class MainNode;
    friend class WorkerNode;

    // Function only call from internal
    static Impl* GetPtr();
};

class MainNode {
public:
    static void SetMainNodeAddress(const std::string& address);

    static bool Start();

    static void Stop();

    static bool WaitClusterReady(size_t numNodes, double timeoutSecond);

    static int64_t NumNodeInCluster();

    static const std::string& GetMainNodeAddress();

    // Function only call from internal
    static core::distributed::MainNode* Get() noexcept;
};

class WorkerNode {
public:
    static int ExecMain(const std::vector<std::string>& arguments);

    static const std::string& GetMainNodeAddress();

    static const std::string& GetWorkerNodeAddress();

    static core::distributed::WorkerNode* Get() noexcept;

private:
    static void Start(const std::string& mainNodeAddress, const std::string& workerNodeAddress, double timeoutSecond);

    static void WaitUntilGetDestroySignal();
};

}  // namespace distributed
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
