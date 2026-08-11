/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "thread_group_same_device_backend.h"

#include <torch/torch.h>

#include "dtorch/common/config.h"
#if DTORCH_WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

#include "dtorch/common/debug.h"
#include "dtorch/core/kernel_stream/kernel_stream.h"
#include "dtorch/external/boost/boost_interprocess.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {
namespace communication {

class IsSameProcessChecker {
public:
    IsSameProcessChecker(const std::string& shmFileName, int rank, int size)
        : mShmFileName(shmFileName),
          mMemory(::boost::interprocess::open_or_create, shmFileName.c_str(), 20480),
          mIsSameProcess(true),
          mRank(rank),
          mSize(size) {
        int64_t currentPid = static_cast<int64_t>(getpid());
        mMemory.Construct<int64_t>(std::to_string(mRank), currentPid);
        for (int i = 0; i < mSize; i++) {
            while (!mMemory.Count<int64_t>(std::to_string(i))) {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(200us);
            }

            int64_t pid = *mMemory.Find<int64_t>(std::to_string(i));
            if (pid != currentPid) {
                mIsSameProcess = false;
                break;
            }
        }
    }

    DTORCH_FORCEINLINE bool IsSameProcess() const noexcept { return mIsSameProcess; }

    ~IsSameProcessChecker() {
        // Wait for all processes to be ready, then remove mShmFileName shared memory file
        mMemory.Construct<int64_t>("barrier" + std::to_string(mRank), 0);
        for (int i = 0; i < mSize; i++) {
            while (!mMemory.Count<int64_t>("barrier" + std::to_string(i))) {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(200us);
            }
        }
        ::boost::interprocess::shared_memory_object::remove(mShmFileName.c_str());
    }

private:
    std::string mShmFileName;
    external::boost::ManagedSharedMemory mMemory;
    bool mIsSameProcess;
    int mRank;
    int mSize;
};

ThreadGroupSameDeviceBackend::ThreadGroupSameDeviceBackend(const std::string& initString, DeviceKind deviceKind,
                                                           int rank, int size)
    : Backend(rank, size), mTensorStore(), mDeviceKind(deviceKind), mRank(rank), mWorldSize(size), mStream(nullptr) {
    DDebugAssert(mWorldSize >= 2);
    InitTensorStore(initString, rank, size);
}

void ThreadGroupSameDeviceBackend::InitTensorStore(const std::string& initString, int rank, int size) {
    TensorStoreConfig config(TensorStoreType::kMemory);

    IsSameProcessChecker checker(initString + "_is_same_process", rank, size);
    if (!checker.IsSameProcess()) {
        config.tensorStoreType = TensorStoreType::kFile;
    }

    mTensorStore = TensorStore::Create(TensorStoreCreateInfo(config, initString, size));
    barrier();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::broadcast(std::vector<at::Tensor>& tensors,
                                                                       const c10d::BroadcastOptions& opts) {
    Sync();
    for (size_t i = 0; i < tensors.size(); i++) {
        int64_t srcRank = opts.rootRank + opts.rootTensor;
        std::string srcRankStr = std::to_string(srcRank);
        DDebugAssert(mStream != nullptr);

        if (mRank == srcRank) {
            mTensorStore->SrcSet(srcRankStr, tensors[i], *mStream, static_cast<size_t>(srcRank - 1));
            mTensorStore->SrcWaitUntilGetFinished(srcRankStr, *mStream);
        } else {
            tensors[i] = mTensorStore->DestGet(srcRankStr, *mStream);
            mTensorStore->DestFinishGet(srcRankStr, *mStream);
        }

        Reset();
    }
    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::allreduce(std::vector<at::Tensor>& tensors,
                                                                       const c10d::AllreduceOptions& opts) {
    Sync();
    for (size_t i = 0; i < tensors.size(); i++) {
        at::Tensor result;
        std::string resultStr = "result";
        std::string rankStr = std::to_string(mRank);
        DDebugAssert(mStream != nullptr);

        mTensorStore->SrcSet(rankStr, tensors[i], *mStream, 1);
        if (mRank == 0) {
            DDebugAssert(opts.reduceOp == c10d::ReduceOp::SUM);
            std::string rankIdxStr = "0";
            result = mTensorStore->DestGet(rankIdxStr, *mStream);
            mTensorStore->DestFinishGet(rankIdxStr, *mStream);
            for (int rankIdx = 1; rankIdx < mWorldSize; rankIdx++) {
                rankIdxStr = std::to_string(rankIdx);
                result += mTensorStore->DestGet(rankIdxStr, *mStream);
                mTensorStore->DestFinishGet(rankIdxStr, *mStream);
            }
        }
        mTensorStore->SrcWaitUntilGetFinished(rankStr, *mStream);

        if (mRank == 0) {
            tensors[i] = result;
            mTensorStore->SrcSet(resultStr, result, *mStream, static_cast<size_t>(mWorldSize - 1));
            mTensorStore->SrcWaitUntilGetFinished(resultStr, *mStream);
        } else {
            tensors[i] = mTensorStore->DestGet(resultStr, *mStream);
            mTensorStore->DestFinishGet(resultStr, *mStream);
        }

        Reset();
    }
    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::reduce(std::vector<at::Tensor>& tensors,
                                                                    const c10d::ReduceOptions& opts) {
    Sync();
    int64_t destRank = opts.rootRank + opts.rootTensor;
    for (size_t i = 0; i < tensors.size(); i++) {
        std::string resultStr = "result";
        std::string rankStr = std::to_string(mRank);
        DDebugAssert(mStream != nullptr);

        mTensorStore->SrcSet(rankStr, tensors[i], *mStream, 1);
        if (mRank == destRank) {
            DDebugAssert(opts.reduceOp == c10d::ReduceOp::SUM);
            std::string rankIdxStr = "0";
            at::Tensor result = mTensorStore->DestGet(rankIdxStr, *mStream);
            mTensorStore->DestFinishGet(rankIdxStr, *mStream);
            for (int rankIdx = 1; rankIdx < mWorldSize; rankIdx++) {
                rankIdxStr = std::to_string(rankIdx);
                result += mTensorStore->DestGet(rankIdxStr, *mStream);
                mTensorStore->DestFinishGet(rankIdxStr, *mStream);
            }
            tensors[i] = result;
        }
        mTensorStore->SrcWaitUntilGetFinished(rankStr, *mStream);

        Reset();
    }
    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
    Sync();
    dtorch::IgnoreUnused(opts);
    DDebugAssert(outputTensors.size() == inputTensors.size());
    for (size_t i = 0; i < inputTensors.size(); i++) {
        at::Tensor result;
        std::string resultStr = "result";
        std::string rankStr = std::to_string(mRank);
        DDebugAssert(mStream != nullptr);

        mTensorStore->SrcSet(rankStr, inputTensors[i], *mStream, static_cast<size_t>(mWorldSize));
        DDebugAssert(outputTensors[i].size() == static_cast<size_t>(mWorldSize));
        for (int rankIdx = 0; rankIdx < mWorldSize; rankIdx++) {
            std::string rankIdxStr = std::to_string(rankIdx);
            outputTensors[i][rankIdx] = mTensorStore->DestGet(rankIdxStr, *mStream);
            mTensorStore->DestFinishGet(rankIdxStr, *mStream);
        }
        mTensorStore->SrcWaitUntilGetFinished(rankStr, *mStream);

        Reset();
    }
    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::_allgather_base(at::Tensor& outputBuffer,
                                                                             at::Tensor& inputBuffer,
                                                                             const c10d::AllgatherOptions& opts) {
    Sync();
    dtorch::IgnoreUnused(opts);
    at::Tensor result;
    std::string resultStr = "result";
    std::string rankStr = std::to_string(mRank);
    DDebugAssert(mStream != nullptr);

    mTensorStore->SrcSet(rankStr, inputBuffer, *mStream, 1);
    if (mRank == 0) {
        std::vector<at::Tensor> allRankInputs;
        for (int rankIdx = 0; rankIdx < mWorldSize; rankIdx++) {
            std::string rankIdxStr = std::to_string(rankIdx);
            allRankInputs.push_back(mTensorStore->DestGet(rankIdxStr, *mStream));
            result = torch::concat(allRankInputs, 0);
            mTensorStore->DestFinishGet(rankIdxStr, *mStream);
        }
    }
    mTensorStore->SrcWaitUntilGetFinished(rankStr, *mStream);

    if (mRank == 0) {
        outputBuffer = result;
        mTensorStore->SrcSet(resultStr, result, *mStream, static_cast<size_t>(mWorldSize - 1));
        mTensorStore->SrcWaitUntilGetFinished(resultStr, *mStream);
    } else {
        outputBuffer = mTensorStore->DestGet(resultStr, *mStream);
        mTensorStore->DestFinishGet(resultStr, *mStream);
    }

    Reset();
    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::_reduce_scatter_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer, const c10d::ReduceScatterOptions& opts) {
    Sync();
    at::Tensor reducedTensor;
    std::string resultStr = "result";
    std::string rankStr = std::to_string(mRank);
    DDebugAssert(mStream != nullptr);

    mTensorStore->SrcSet(rankStr, inputBuffer, *mStream, 1);
    if (mRank == 0) {
        DDebugAssert(opts.reduceOp == c10d::ReduceOp::SUM);
        std::string rankIdxStr = "0";
        reducedTensor = mTensorStore->DestGet(rankIdxStr, *mStream);
        mTensorStore->DestFinishGet(rankIdxStr, *mStream);
        for (int rankIdx = 1; rankIdx < mWorldSize; rankIdx++) {
            rankIdxStr = std::to_string(rankIdx);
            reducedTensor += mTensorStore->DestGet(rankIdxStr, *mStream);
            mTensorStore->DestFinishGet(rankIdxStr, *mStream);
        }
    }
    mTensorStore->SrcWaitUntilGetFinished(rankStr, *mStream);

    if (mRank == 0) {
        mTensorStore->SrcSet(resultStr, reducedTensor, *mStream, static_cast<size_t>(mWorldSize - 1));
        mTensorStore->SrcWaitUntilGetFinished(resultStr, *mStream);
    } else {
        reducedTensor = mTensorStore->DestGet(resultStr, *mStream);
        mTensorStore->DestFinishGet(resultStr, *mStream);
    }

    auto chunkedTensors = torch::chunk(reducedTensor, mWorldSize, 0);
    DAlwaysAssert(chunkedTensors.size() == static_cast<size_t>(mWorldSize));
    outputBuffer = chunkedTensors[mRank];

    Reset();
    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::alltoall(std::vector<at::Tensor>& outputTensors,
                                                                      std::vector<at::Tensor>& inputTensors,
                                                                      const c10d::AllToAllOptions& opts) {
    dtorch::IgnoreUnused(opts);
    Sync();
    DDebugAssert(mStream != nullptr);
    DDebugAssert(inputTensors.size() == static_cast<size_t>(mWorldSize));
    DDebugAssert(outputTensors.size() == static_cast<size_t>(mWorldSize));
    for (int idx = 0; idx < mWorldSize; idx++) {
        std::string keyStr = std::to_string(mRank) + "_" + std::to_string(idx);
        mTensorStore->SrcSet(keyStr, inputTensors[idx], *mStream, 1);
    }

    for (int idx = 0; idx < mWorldSize; idx++) {
        std::string keyStr = std::to_string(idx) + "_" + std::to_string(mRank);
        outputTensors[idx] = mTensorStore->DestGet(keyStr, *mStream);
        mTensorStore->DestFinishGet(keyStr, *mStream);
    }

    for (int idx = 0; idx < mWorldSize; idx++) {
        std::string keyStr = std::to_string(mRank) + "_" + std::to_string(idx);
        mTensorStore->SrcWaitUntilGetFinished(keyStr, *mStream);
    }

    Reset();
    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::barrier(const c10d::BarrierOptions& opts) {
    dtorch::IgnoreUnused(opts);

    mTensorStore->Barrier();

    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::send(std::vector<at::Tensor>& tensors, int dstRank,
                                                                  int tag) {
    Sync();
    std::string tagStr = std::to_string(tag);
    DDebugAssert(mStream != nullptr);
    DAlwaysAssert(mRank == dstRank);

    for (size_t i = 0; i < tensors.size(); i++) {
        mTensorStore->SrcSet(tagStr, tensors[i], *mStream, 1);
        mTensorStore->SrcWaitUntilGetFinished(tagStr, *mStream);
        Reset();
    }

    return c10::make_intrusive<WorkSameDevice>();
}

c10::intrusive_ptr<c10d::Work> ThreadGroupSameDeviceBackend::recv(std::vector<at::Tensor>& tensors, int srcRank,
                                                                  int tag) {
    Sync();
    std::string tagStr = std::to_string(tag);
    DDebugAssert(mStream != nullptr);
    DAlwaysAssert(mRank == srcRank);

    for (size_t i = 0; i < tensors.size(); i++) {
        mTensorStore->DestGet(tagStr, *mStream);
        mTensorStore->DestFinishGet(tagStr, *mStream);
        Reset();
    }

    return c10::make_intrusive<WorkSameDevice>();
}

void ThreadGroupSameDeviceBackend::Sync() {
    IgnoreUnused(mDeviceKind);
#if DTORCH_WITH_CUDA
    if (mDeviceKind == DeviceKind::kGpu) {
        DAlwaysAssert(torch::cuda::is_available());
        at::cuda::getCurrentCUDAStream().synchronize();
    }
#endif
}

void ThreadGroupSameDeviceBackend::SetStream(DeviceStream& stream) { mStream = &stream; }

void ThreadGroupSameDeviceBackend::Reset() {
    barrier();
    mTensorStore->Reset();
    barrier();
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
