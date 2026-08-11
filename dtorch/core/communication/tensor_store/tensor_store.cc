/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "tensor_store.h"

#include <torch/torch.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"
#include "dtorch/external/device/device_event.h"
#include "file_tensor_store.h"
#include "memory_tensor_store.h"
#include "network_tensor_store.h"

namespace dtorch {
namespace core {
namespace communication {

std::shared_ptr<TensorStore> TensorStore::Create(const TensorStoreCreateInfo& createInfo) {
    switch (createInfo.tensorStoreType) {
        case TensorStoreType::kMemory:
            return std::make_shared<MemoryTensorStore>(createInfo.storeKey, createInfo.worldSize);
        case TensorStoreType::kFile:
            return std::make_shared<FileTensorStore>(createInfo.storeKey, createInfo.worldSize);
        // case TensorStoreType::kNetwork:
        //     return std::make_shared<NetworkTensorStore>(createInfo.networkAddress, createInfo.worldSize);
        default:
            DLogError() << "Unsupport tensor store type: " << EnumAsInteger(createInfo.tensorStoreType);
            DUnimplemented();
            break;
    }

    return nullptr;
}

torch::Tensor TensorStore::DestGetAndToDevice(const std::string& key, DeviceStream& stream,
                                              const Device& targetDevice) {
    torch::Tensor tensor = this->DestGet(key, stream);

    // Cross-stream synchronization for GPU-to-GPU transfer:
    // DestGet() returns a tensor residing on the source GPU, and tensor.to(targetDevice) will
    // execute on the source GPU's current stream (srcDeviceStreamOfThisThread). However, the
    // `stream` parameter passed to DestGetAndToDevice is the destination GPU's stream, and any
    // subsequent ops on that stream must be ordered after tensor.to() completes. We therefore
    // record an event on the caller's (dest) stream and make the source GPU stream wait on it,
    // so tensor.to() does not start until all prior work on the dest stream has finished.
    auto event = external::device::DeviceEvent::CreateDeviceEvent(stream.GetDeviceKind());
    event->Record(stream);
    DeviceStream srcDeviceStreamOfThisThread =
        DeviceStream::GetCurrentStream(external::torch::TorchUtil::GetDevice(tensor));
    event->StreamWaitEvent(srcDeviceStreamOfThisThread);

    auto options = torch::TensorOptions();
    options = options.device(external::torch::TorchUtil::ToDevice(targetDevice));
    auto result = tensor.to(options);
    this->DestFinishGet(key, stream);
    return result;
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
