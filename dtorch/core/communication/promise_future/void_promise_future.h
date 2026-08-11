/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "dtorch/common/utilities.h"

namespace dtorch {
namespace core {
namespace communication {

// ============================================================
// VoidPromiseType enum
// ============================================================

enum class VoidPromiseType {
    kMemory = 0,  // Single-process multi-thread, uses std::promise / std::future
    kFile,        // Single-machine multi-process, uses Boost IPC shared memory
    kNetwork,     // Multi-machine multi-process (reserved)
};

// ============================================================
// class VoidFuture — Future abstract base class
// ============================================================

class VoidFuture {
public:
    VoidFuture() = default;

    virtual ~VoidFuture() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(VoidFuture);

    // Blocking, consuming (like std::future::get)
    virtual void Get() = 0;

    // Blocking, non-consuming (like std::future::wait)
    virtual void Wait() = 0;

    // Wait with timeout. Returns:
    //   true  → promise fulfilled
    //   false → timeout, not yet fulfilled
    virtual bool WaitFor(int64_t timeoutMs) = 0;

    // Check if the promise has been fulfilled (non-blocking).
    virtual bool IsReady() const = 0;
};

// ============================================================
// class VoidPromise — Promise abstract base class
// ============================================================

class VoidPromise {
public:
    VoidPromise() = default;

    virtual ~VoidPromise() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(VoidPromise);

    // ---------- Core interface ----------

    // Signal completion (called by SyncKernel::Compute())
    virtual void SetValue() = 0;

    // Get associated Future (called by Client, only once)
    virtual std::unique_ptr<VoidFuture> GetFuture() = 0;

    // Return current Promise type
    virtual VoidPromiseType GetType() const = 0;

    // ---------- Serialization ----------

    // Serialize to string (Memory mode returns empty, File mode returns shared memory filename)
    virtual std::string Serialize() const = 0;

    // Reconstruct from serialized string and attach to existing shared memory.
    // Called by CreateVoidPromiseFromSerialized() factory function.
    virtual void Deserialize(const std::string& data) = 0;
};

// ============================================================
// Factory functions
// ============================================================

// Create a new Promise (used by Client to initiate async sync)
std::unique_ptr<VoidPromise> CreateVoidPromise(VoidPromiseType type);

// Reconstruct Promise from serialized data (used during deserialization on Worker side).
// Internally creates the appropriate Promise type, then calls Deserialize(data).
std::unique_ptr<VoidPromise> CreateVoidPromiseFromSerialized(VoidPromiseType type, const std::string& data);

// Determine the Promise type to use.
// kFile when perDevicePerProcess is true, kMemory otherwise.
VoidPromiseType GetVoidPromiseType(bool perDevicePerProcess);

}  // namespace communication
}  // namespace core
}  // namespace dtorch
