/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <optional>
#include <string>

#include <torch/torch.h>

#include "dtorch/common/utilities.h"

namespace dtorch {
namespace core {
class Operand;
}  // namespace core
}  // namespace dtorch

namespace dtorch {
namespace core {
namespace communication {

// ============================================================
// TensorPromiseType enum
// ============================================================

enum class TensorPromiseType {
    kMemory = 0,  // Single-process multi-thread, uses std::promise / std::future
    kFile,        // Single-machine multi-process, uses Boost IPC shared memory
    kNetwork,     // Multi-machine multi-process (reserved)
};

// ============================================================
// class TensorFuture — Future abstract base class
// ============================================================

class TensorFuture {
public:
    TensorFuture() = default;

    virtual ~TensorFuture() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(TensorFuture);

    // Blocking wait (waits indefinitely until promise is set)
    virtual std::shared_ptr<torch::Tensor> Get() = 0;

    // Blocking wait (like std::future::wait, non-consuming, returns void)
    virtual void Wait() = 0;

    // Wait with timeout. Returns true if ready, false on timeout.
    virtual bool WaitFor(int64_t timeoutMs) = 0;

    // Check if the value is actually ready (non-blocking).
    virtual bool IsReady() const = 0;
};

// ============================================================
// class TensorPromise — Promise abstract base class
// ============================================================

class TensorPromise {
public:
    TensorPromise() = default;

    virtual ~TensorPromise() = default;

    DTORCH_DISABLE_COPY_AND_MOVE(TensorPromise);

    // ---------- Core interface ----------

    // Write Tensor value (called by GetTensorOp::Compute())
    virtual void SetValue(std::shared_ptr<torch::Tensor> tensor) = 0;

    // Get associated Future (called by Client, only once)
    virtual std::unique_ptr<TensorFuture> GetFuture() = 0;

    // Return current Promise type
    virtual TensorPromiseType GetType() const = 0;

    // ---------- Serialization ----------

    // Serialize to string (Memory mode returns empty, File mode returns shared memory filename)
    virtual std::string Serialize() const = 0;

    // Reconstruct from serialized string and attach to existing shared memory.
    // Called by CreateTensorPromiseFromSerialized() factory function.
    virtual void Deserialize(const std::string& data) = 0;
};

// ============================================================
// Factory functions
// ============================================================

// Create a new Promise (used by Client to initiate async get)
std::unique_ptr<TensorPromise> CreateTensorPromise(TensorPromiseType type);

// Reconstruct Promise from serialized data (used during deserialization on Worker side).
// Internally creates the appropriate Promise type, then calls Deserialize(data).
std::unique_ptr<TensorPromise> CreateTensorPromiseFromSerialized(TensorPromiseType type, const std::string& data);

// Determine the Promise type to use.
// CPU tensors: always kMemory (in-process NaiveRunner even with perDevicePerProcess).
// GPU tensors: kFile when perDevicePerProcess is true, kMemory otherwise.
TensorPromiseType GetTensorPromiseTypeFromOperand(const core::Operand& operand, bool perDevicePerProcess);

}  // namespace communication
}  // namespace core
}  // namespace dtorch
