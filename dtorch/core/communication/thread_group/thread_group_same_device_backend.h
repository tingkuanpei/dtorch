/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>

#include "dtorch/common/utilities.h"
#include "dtorch/core/communication/tensor_store/tensor_store.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace core {
namespace communication {

class ThreadGroupSameDeviceBackend : public c10d::Backend {
public:
    ThreadGroupSameDeviceBackend(const std::string& initString, DeviceKind deviceKind, int rank, int size);

    DTORCH_DISABLE_COPY_AND_MOVE(ThreadGroupSameDeviceBackend);

    virtual ~ThreadGroupSameDeviceBackend() override = default;

    const std::string getBackendName() const override { return "same_device"; }

    c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor>& tensors,
                                             const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

    c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor>& tensors,
                                             const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> reduce(std::vector<at::Tensor>& tensors,
                                          const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                             std::vector<at::Tensor>& inputTensors,
                                             const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> _allgather_base(
        at::Tensor& outputBuffer, at::Tensor& inputBuffer,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
        at::Tensor& outputBuffer, at::Tensor& inputBuffer,
        const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

    // c10::intrusive_ptr<c10d::Work> reduce_scatter(std::vector<at::Tensor>& outputTensors,
    //                                         std::vector<std::vector<at::Tensor>>& inputTensors,
    //                                         const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions())
    //                                         override;

    c10::intrusive_ptr<c10d::Work> alltoall(std::vector<at::Tensor>& outputTensors,
                                            std::vector<at::Tensor>& inputTensors,
                                            const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

    c10::intrusive_ptr<c10d::Work> barrier(const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

    c10::intrusive_ptr<c10d::Work> send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override;

    c10::intrusive_ptr<c10d::Work> recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override;

    // c10::intrusive_ptr<c10d::Work> gather(std::vector<std::vector<at::Tensor>>& outputTensors,
    //                                 std::vector<at::Tensor>& inputTensors,
    //                                 const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

    // c10::intrusive_ptr<c10d::Work> scatter(std::vector<at::Tensor>& outputTensors,
    //                                  std::vector<std::vector<at::Tensor>>& inputTensors,
    //                                  const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

    void Sync();

    void SetStream(DeviceStream& stream);

private:
    void InitTensorStore(const std::string& initString, int rank, int size);

    void Reset();

public:
    class WorkSameDevice : public c10d::Work {
    public:
        WorkSameDevice() : Work() { finish(); }
    };

private:
    std::shared_ptr<TensorStore> mTensorStore;
    DeviceKind mDeviceKind;
    int mRank;
    int mWorldSize;
    DeviceStream* mStream;
};

}  // namespace communication
}  // namespace core
}  // namespace dtorch
