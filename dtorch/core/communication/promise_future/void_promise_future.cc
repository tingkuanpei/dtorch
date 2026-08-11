/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "void_promise_future.h"

#include "file_void_promise_future.h"
#include "memory_void_promise_future.h"

namespace dtorch {
namespace core {
namespace communication {

std::unique_ptr<VoidPromise> CreateVoidPromise(VoidPromiseType type) {
    switch (type) {
        case VoidPromiseType::kMemory:
            return std::make_unique<MemoryVoidPromise>();
        case VoidPromiseType::kFile:
            return std::make_unique<FileVoidPromise>();
        case VoidPromiseType::kNetwork:
            throw std::invalid_argument("NetworkVoidPromise is reserved and not yet implemented");
    }
    return nullptr;
}

std::unique_ptr<VoidPromise> CreateVoidPromiseFromSerialized(VoidPromiseType type, const std::string& data) {
    switch (type) {
        case VoidPromiseType::kMemory:
            // Memory mode: no serialization needed in-process. The promise is created
            // fresh on the remote side (not reached in practice for kMemory).
            return nullptr;
        case VoidPromiseType::kFile: {
            auto promise = std::make_unique<FileVoidPromise>();
            promise->Deserialize(data);
            return promise;
        }
        case VoidPromiseType::kNetwork:
            throw std::invalid_argument("NetworkVoidPromise is reserved and not yet implemented");
    }
    return nullptr;
}

VoidPromiseType GetVoidPromiseType(bool perDevicePerProcess) {
    return perDevicePerProcess ? VoidPromiseType::kFile : VoidPromiseType::kMemory;
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
