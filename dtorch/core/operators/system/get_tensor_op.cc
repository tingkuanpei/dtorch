/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "get_tensor_op.h"

#include <torch/cuda.h>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/communication/promise_future/tensor_promise_future.h"
#include "dtorch/core/type.h"
#include "dtorch/external/boost/boost_asio_thread_pool.h"
#include "dtorch/external/device/device_event.h"
#include "dtorch/external/device/device_stream.h"

namespace dtorch {
namespace core {

void GetTensorOp::Compute(const TorchTensorOptArray& inputs, TorchTensorArray& outputs) const {
    DDebugAssert(outputs.size() == 0);

    const auto& param = GetOpParam<GetTensorParam>();
    DAlwaysAssert(param.promise);
    // NullTensor can't get tensor
    DAlwaysAssert(inputs[0].has_value());

    if (inputs[0].value().is_cuda()) {
        // Record an event on the current stream so we can wait for
        // only this stream's work to complete (not the entire device).
        Device localDevice(DeviceKind::kGpu, inputs[0].value().get_device());
        auto stream = external::device::DeviceStream::GetCurrentStream(localDevice);
        auto event = external::device::DeviceEvent::CreateDeviceEvent(localDevice.deviceKind);
        event->Record(stream);

        // Transfer ownership of tensor and promise to a background thread.
        // The thread pool polls the event; once ready, it calls SetValue
        // on a regular CPU thread where CUDA IPC functions are permitted.
        auto tensor = std::make_shared<torch::Tensor>(inputs[0].value());
        auto promise = std::move(param.promise);

        external::boost::BoostAsioThreadPool::GetInstance().Post(
            [event = std::move(event), tensor, promise = std::move(promise)]() {
                event->Synchronize();
                promise->SetValue(tensor);
            });
    } else {
        param.promise->SetValue(std::make_shared<torch::Tensor>(inputs[0].value()));
    }
}

std::string GetTensorOp::GetDescribeString() const { return "GetTensorOp"; }

}  // namespace core
}  // namespace dtorch
