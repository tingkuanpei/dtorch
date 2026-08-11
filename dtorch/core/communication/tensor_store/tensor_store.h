/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <vector>

#include "dtorch/core/type.h"
#include "dtorch/external/device/device_stream.h"
#include "dtorch/external/torch/torch_util.h"

using dtorch::external::device::DeviceStream;

namespace dtorch {
namespace core {
namespace communication {

// Backend type for TensorStore. Determines the underlying transport mechanism.
//   kMemory  — In-process, thread-safe tensor exchange via std::mutex + std::condition_variable.
//              All participants must live in the same process. Fastest option.
//   kFile    — Inter-process tensor exchange via Boost shared memory. Supports GPU tensors
//              through CUDA IPC (cross-process) and same-process fast path.
//   kNetwork — Inter-machine tensor exchange (reserved, not yet implemented).
enum class TensorStoreType {
    kMemory = 0,
    kFile,
    kNetwork,
    kCount,
};

// Configuration struct that selects the TensorStore backend.
struct TensorStoreConfig {
    TensorStoreType tensorStoreType;

public:
    TensorStoreConfig(TensorStoreType tensorStoreType) : tensorStoreType(tensorStoreType) {}
};

// Creation parameters for TensorStore::Create().
//   storeKey  — Unique identifier shared by all participants. Threads/processes that use the
//               same storeKey join the same tensor exchange session (e.g. the same logical graph).
//   worldSize — Total number of participants that will call Barrier(). Must be identical across
//               all participants sharing the same storeKey.
struct TensorStoreCreateInfo {
    TensorStoreType tensorStoreType;
    std::string storeKey;
    int worldSize;

public:
    TensorStoreCreateInfo(const TensorStoreConfig& config, const std::string& storeKey, int worldSize)
        : tensorStoreType(config.tensorStoreType), storeKey(storeKey), worldSize(worldSize) {}
};

// TensorStore is a producer-consumer tensor exchange abstraction that lets multiple
// threads/processes share tensors with synchronized access.
//
// Protocol overview — every tensor exchange follows a four-step handshake:
//
//   Producer (1 writer):                     Consumers (N readers, where N = getCount):
//   1. SrcSet(key, tensor, getCount=N)       3. DestGet(key)          — blocks if tensor not yet set
//   2. SrcWaitUntilGetFinished(key)           4. DestFinishGet(key)    — signals completion to producer
//
//   The producer blocks at step 2 until all N consumers have called DestFinishGet (step 4).
//   Consumers block at step 3 until the producer has called SrcSet (step 1).
//
// Thread safety: all methods may be called concurrently from different threads/processes.
// The underlying implementation uses mutex + condition_variable (kMemory) or interprocess
// mutex + condition_variable (kFile).
//
// Lifecycle:
//   auto store = TensorStore::Create(createInfo);  // constructor calls Barrier() internally
//   // ... producer/consumer operations ...
//   store->Reset();  // clear all tensors and reset state (barrier-synchronized)
//   // store is destroyed (destructor)
//
// Usage example:
//
//   // --- Setup: 2 threads sharing the same storeKey ---
//   TensorStoreConfig config{TensorStoreType::kMemory};
//   TensorStoreCreateInfo info{config, "my_graph_store", /*worldSize=*/2};
//
//   // --- Producer thread ---
//   auto store = TensorStore::Create(info);
//   DeviceStream stream = ...;
//   store->SrcSet("layer1_out", myTensor, stream, /*getCount=*/1);
//   store->SrcWaitUntilGetFinished("layer1_out", stream);
//   // Safe to reuse/release myTensor after this point.
//
//   // --- Consumer thread ---
//   auto store = TensorStore::Create(info);
//   DeviceStream stream = ...;
//   torch::Tensor result = store->DestGet("layer1_out", stream);
//   // ... use result ...
//   store->DestFinishGet("layer1_out", stream);
//
//   // Or use the convenience helper:
//   torch::Tensor result = store->DestGetAndToDevice("layer1_out", stream, targetDevice);
class TensorStore {
public:
    // Factory method. Creates a concrete TensorStore instance based on createInfo.tensorStoreType.
    // The constructor of each implementation calls Barrier() internally, so all participants
    // must call Create() concurrently (or within a short window) to avoid deadlock.
    static std::shared_ptr<TensorStore> Create(const TensorStoreCreateInfo& createInfo);

public:
    TensorStore() = default;

    virtual ~TensorStore() = default;

    // --- Producer-side API ---

    // Publish a tensor under the given key so that consumers can retrieve it via DestGet().
    //   key             — Unique name for this tensor within the store session.
    //   value           — The tensor to share. Must reside on the device matching stream.
    //   stream          — The CUDA/CPU stream on which value was produced. An internal event
    //                     is recorded on this stream so consumers synchronize correctly.
    //   getCount        — Number of DestGet() calls expected before the producer is unblocked.
    //   destGetDeviceKind — Optional hint for cross-device transfer optimization. When set and
    //                     different from the source device kind, the tensor is moved to the
    //                     destination device kind before storing (e.g. CPU→GPU to enable CUDA IPC
    //                     in FileTensorStore). For GPU tensors, CUDA IPC / NCCL send/recv provide
    //                     better performance than CPU serialization/deserialization.
    virtual void SrcSet(const std::string& key, const torch::Tensor& value, DeviceStream& stream, size_t getCount,
                        std::optional<DeviceKind> destGetDeviceKind = std::nullopt) = 0;

    // Block the calling thread until all getCount consumers have called DestFinishGet() for the
    // given key. After this returns, the producer may safely reuse or release the tensor memory.
    // The stream parameter is unused in current implementations but reserved for future use
    // (e.g. stream-based synchronization).
    virtual void SrcWaitUntilGetFinished(const std::string& key, DeviceStream& stream) = 0;

    // --- Consumer-side API ---

    // Retrieve the tensor associated with key. Blocks until the producer calls SrcSet() for
    // this key. The returned tensor is synchronized with the producer's stream via an internal
    // event recorded during SrcSet(), so it is safe to use on the calling stream immediately.
    // Multiple consumers may call DestGet() concurrently on the same key.
    virtual torch::Tensor DestGet(const std::string& key, DeviceStream& stream) = 0;

    // Signal that this consumer has finished using the tensor. Must be called exactly once
    // per DestGet() call. When all getCount consumers have called DestFinishGet(), the producer
    // is unblocked from SrcWaitUntilGetFinished().
    // For GPU tensors, this also calls c10::cuda::CUDACachingAllocator::recordStream() on the
    // source tensor, preventing premature memory reuse before the consumer stream completes.
    virtual void DestFinishGet(const std::string& key, DeviceStream& stream) = 0;

    // Clear all stored tensors and reset internal state. This is a barrier-synchronized
    // operation: it calls Barrier() both before and after clearing, so all participants
    // must call Reset() concurrently.
    virtual void Reset() = 0;

    // Synchronization barrier. Blocks until all worldSize participants have called Barrier().
    // Internally used by the constructor and Reset() to ensure all participants start from
    // a consistent state.
    virtual void Barrier() = 0;

    // Convenience helper: calls DestGet() and then moves the returned tensor to targetDevice.
    // Handles cross-stream synchronization — records an event on the DestGet stream and waits
    // on the source tensor's current stream before calling tensor.to(device).
    torch::Tensor DestGetAndToDevice(const std::string& key, DeviceStream& stream, const Device& targetDevice);
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
