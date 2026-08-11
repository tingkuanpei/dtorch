/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "sync_kernel.h"

#include "dtorch/common/debug.h"
#include "dtorch/core/operators/system/sync_op.h"
#include "dtorch/external/boost/boost_asio_thread_pool.h"
#include "dtorch/external/device/device_event.h"
#include "dtorch/external/device/device_stream.h"

namespace dtorch {
namespace core {

void SyncKernel::Compute(const TorchTensorOptArray& /*inputs*/, TorchTensorArray& /*outputs*/) {
    auto& param = GetOpParam<SyncParam>();
    DAlwaysAssert(param.syncDevices.size() == mNumKernelForThisOp);

    const Device& globalDevice = GetGlobalDevice();

    // Find which device index this kernel is responsible for
    int deviceIdx = -1;
    for (size_t i = 0; i < param.syncDevices.size(); i++) {
        if (param.syncDevices[i] == globalDevice) {
            deviceIdx = static_cast<int>(i);
            break;
        }
    }
    DAlwaysAssert(deviceIdx >= 0 && static_cast<size_t>(deviceIdx) < param.promises.size());
    DAlwaysAssert(param.promises[deviceIdx] != nullptr);

    const Device& localDevice = GetLocalDevice();
    if (localDevice.deviceKind == DeviceKind::kGpu) {
        // Record a CUDA event on the current stream so we can wait for
        // only this stream's work to complete (not the entire device).
        auto stream = external::device::DeviceStream::GetCurrentStream(localDevice);
        auto event = external::device::DeviceEvent::CreateDeviceEvent(localDevice.deviceKind);
        event->Record(stream);

        // Transfer ownership of the promise to a background thread.
        // The thread pool polls the event; once ready, it calls SetValue()
        // on a regular CPU thread.
        auto promise = std::move(param.promises[deviceIdx]);

        external::boost::BoostAsioThreadPool::GetInstance().Post(
            [event = std::move(event), promise = std::move(promise)]() {
                event->Synchronize();
                promise->SetValue();
            });
    } else {
        // CPU: kernel execution is synchronous — set the promise directly
        param.promises[deviceIdx]->SetValue();
    }
}

}  // namespace core
}  // namespace dtorch
