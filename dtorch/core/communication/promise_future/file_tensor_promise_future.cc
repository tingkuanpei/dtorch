/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "file_tensor_promise_future.h"

#include <chrono>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "dtorch/common/debug.h"
#include "dtorch/core/global_id_manager.h"
#include "dtorch/external/boost/boost_interprocess.h"
#include "dtorch/external/torch/torch_util.h"

namespace dtorch {
namespace core {
namespace communication {

// ============================================================
// Shared memory structure — used by both FileTensorPromise and FileTensorFuture
// ============================================================

struct FileTensorPromiseFutureShmImpl {
    std::unique_ptr<external::boost::ManagedSharedMemory> sharedMemory;
    external::boost::InterprocessMutex* mutex;
    external::boost::InterprocessCondition* cond;
    bool* hasValue;

    FileTensorPromiseFutureShmImpl() : sharedMemory(), mutex(nullptr), cond(nullptr), hasValue(nullptr) {}

    DTORCH_DISABLE_COPY_AND_MOVE(FileTensorPromiseFutureShmImpl);

    void Create(const std::string& fileName) {
        sharedMemory = std::make_unique<external::boost::ManagedSharedMemory>(::boost::interprocess::create_only,
                                                                              fileName.c_str(), 65536);
        mutex = sharedMemory->FindOrConstruct<external::boost::InterprocessMutex>("Mutex");
        cond = sharedMemory->FindOrConstruct<external::boost::InterprocessCondition>("Cond");
        hasValue = sharedMemory->FindOrConstruct<bool>("HasValue", false);
    }

    void Open(const std::string& fileName) {
        sharedMemory =
            std::make_unique<external::boost::ManagedSharedMemory>(::boost::interprocess::open_only, fileName.c_str());
        mutex = sharedMemory->Find<external::boost::InterprocessMutex>("Mutex");
        cond = sharedMemory->Find<external::boost::InterprocessCondition>("Cond");
        hasValue = sharedMemory->Find<bool>("HasValue");
    }
};

// ============================================================
// Static helper to generate unique shared memory filename
// ============================================================

static std::string GenerateShmFileName() {
    uint64_t id = TensorPromiseIdManager::GetSingleton().GetUniqueId();
    return external::boost::ManagedSharedMemory::GetShmFileNameWithPrefix("tensor_promise_" + std::to_string(id) + "_" +
                                                                          std::to_string(getpid()));
}

// ============================================================
// FileTensorFuture
// ============================================================

FileTensorFuture::FileTensorFuture(const std::string& shmFileName)
    : TensorFuture(), mShmImpl(std::make_shared<FileTensorPromiseFutureShmImpl>()), mValueConsumed(false) {
    mShmImpl->Open(shmFileName);
}

FileTensorFuture::~FileTensorFuture() = default;

void FileTensorFuture::Wait() {
    DDebugAssert(mShmImpl != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);
    if (!*mShmImpl->hasValue) {
        mShmImpl->cond->wait(lock, [&] { return *mShmImpl->hasValue; });
    }
    DDebugAssert(*mShmImpl->hasValue);
}

std::shared_ptr<torch::Tensor> FileTensorFuture::Get() {
    DDebugAssert(mShmImpl != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);
    if (!*mShmImpl->hasValue) {
        mShmImpl->cond->wait(lock, [&] { return *mShmImpl->hasValue; });
    }
    DDebugAssert(*mShmImpl->hasValue);

    std::string tensorData = mShmImpl->sharedMemory->FindStr("TensorData");
    mValueConsumed = true;
    lock.unlock();

    return std::make_shared<torch::Tensor>(external::torch::TorchUtil::FromIpcMemHandle(tensorData));
}

bool FileTensorFuture::WaitFor(int64_t timeoutMs) {
    DDebugAssert(mShmImpl != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);
    if (!*mShmImpl->hasValue) {
        auto absTime = boost::posix_time::microsec_clock::universal_time() + boost::posix_time::milliseconds(timeoutMs);
        if (!mShmImpl->cond->timed_wait(lock, absTime, [&] { return *mShmImpl->hasValue; })) {
            return false;
        }
    }
    return true;
}

bool FileTensorFuture::IsReady() const {
    // Non-blocking check: the value is ready if hasValue is set and not yet consumed.
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);
    return *mShmImpl->hasValue && !mValueConsumed;
}

// ============================================================
// FileTensorPromise
// ============================================================

FileTensorPromise::FileTensorPromise()
    : TensorPromise(), mShmFileName(), mIsCreator(false), mFutureTaken(false), mShmImpl(nullptr) {
    CreateSharedMemory();
}

FileTensorPromise::~FileTensorPromise() {
    // Note: shared memory file is intentionally NOT removed here.
    // The FileTensorFuture holds a persistent open connection and will
    // clean up when it is destroyed and the last reference is released.
    // In cross-process scenarios, this leaves a small temporary file
    // which is acceptable (persisted in /dev/shm which is tmpfs).
}

void FileTensorPromise::CreateSharedMemory() {
    mShmFileName = GenerateShmFileName();
    mIsCreator = true;
    mShmImpl = std::make_shared<FileTensorPromiseFutureShmImpl>();
    mShmImpl->Create(mShmFileName);
}

void FileTensorPromise::OpenSharedMemory() {
    mIsCreator = false;
    mShmImpl = std::make_shared<FileTensorPromiseFutureShmImpl>();
    mShmImpl->Open(mShmFileName);
}

void FileTensorPromise::SetValue(std::shared_ptr<torch::Tensor> tensor) {
    DDebugAssert(mShmImpl != nullptr);
    DDebugAssert(mShmImpl->mutex != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);

    DDebugAssert(!(*mShmImpl->hasValue) && "SetValue should only be called once");

    std::string tensorData = external::torch::TorchUtil::ToIpcMemHandle(*tensor);
    mShmImpl->sharedMemory->ConstructString("TensorData", tensorData);
    *mShmImpl->hasValue = true;

    lock.unlock();
    mShmImpl->cond->notify_all();
}

std::unique_ptr<TensorFuture> FileTensorPromise::GetFuture() {
    if (mFutureTaken) {
        DLogFatal() << "FileTensorPromise::GetFuture() can only be called once.";
        return nullptr;
    }
    mFutureTaken = true;
    return std::make_unique<FileTensorFuture>(mShmFileName);
}

std::string FileTensorPromise::Serialize() const { return mShmFileName; }

void FileTensorPromise::Deserialize(const std::string& data) {
    mShmFileName = data;
    OpenSharedMemory();
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
