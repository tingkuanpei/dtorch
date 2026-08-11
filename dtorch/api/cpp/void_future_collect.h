/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>

namespace dtorch {
namespace core {
namespace communication {
class VoidFuture;
}  // namespace communication
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace api {
namespace cpp {

// ============================================================
// class VoidFutureCollect
// ============================================================
//
// Collects multiple VoidFuture instances and waits for all of them.
// Used as the return type of Graph::Sync(), which creates one
// VoidFuture per device being synchronized.
//
// Usage:
//   auto futures = graph.Sync();
//   futures.Wait();  // blocks until all devices are synced

class VoidFutureCollect {
public:
    VoidFutureCollect();

    ~VoidFutureCollect();

    // Add a future to the collection
    void AddFuture(std::unique_ptr<core::communication::VoidFuture> future);

    // Blocking, consuming: get all futures (like std::future::get)
    void Get();

    // Blocking, non-consuming: wait for all futures (like std::future::wait)
    void Wait();

    // Non-blocking check: are all futures ready?
    bool IsReady() const;

private:
    struct Impl;
    std::shared_ptr<Impl> mImplPtr;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
