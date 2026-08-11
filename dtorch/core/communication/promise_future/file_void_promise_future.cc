/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "file_void_promise_future.h"

#include <chrono>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "dtorch/common/debug.h"
#include "dtorch/core/global_id_manager.h"
#include "dtorch/external/boost/boost_interprocess.h"

namespace dtorch {
namespace core {
namespace communication {

// ============================================================
// Shared memory structure — used by both FileVoidPromise and FileVoidFuture
// ============================================================

struct VoidPromiseFutureShmImpl {
    std::unique_ptr<external::boost::ManagedSharedMemory> sharedMemory;
    external::boost::InterprocessMutex* mutex;
    external::boost::InterprocessCondition* cond;
    bool* hasValue;

    VoidPromiseFutureShmImpl() : sharedMemory(), mutex(nullptr), cond(nullptr), hasValue(nullptr) {}

    DTORCH_DISABLE_COPY_AND_MOVE(VoidPromiseFutureShmImpl);

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
// VoidPromiseIdManager — unique ID generator for shared memory filenames
// ============================================================

using VoidPromiseIdManager = GlobalIdManager<3>;

// ============================================================
// Static helper to generate unique shared memory filename
// ============================================================

static std::string GenerateShmFileName() {
    uint64_t id = VoidPromiseIdManager::GetSingleton().GetUniqueId();
    return external::boost::ManagedSharedMemory::GetShmFileNameWithPrefix("void_promise_" + std::to_string(id) + "_" +
                                                                          std::to_string(getpid()));
}

// ============================================================
// FileVoidFuture
// ============================================================

FileVoidFuture::FileVoidFuture(const std::string& shmFileName)
    : VoidFuture(), mShmImpl(std::make_shared<VoidPromiseFutureShmImpl>()), mValueConsumed(false) {
    mShmImpl->Open(shmFileName);
}

FileVoidFuture::~FileVoidFuture() = default;

void FileVoidFuture::Get() {
    DDebugAssert(mShmImpl != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);
    if (!*mShmImpl->hasValue) {
        mShmImpl->cond->wait(lock, [&] { return *mShmImpl->hasValue; });
    }
    DDebugAssert(*mShmImpl->hasValue);
    mValueConsumed = true;
}

void FileVoidFuture::Wait() {
    DDebugAssert(mShmImpl != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);
    if (!*mShmImpl->hasValue) {
        mShmImpl->cond->wait(lock, [&] { return *mShmImpl->hasValue; });
    }
    DDebugAssert(*mShmImpl->hasValue);
}

bool FileVoidFuture::WaitFor(int64_t timeoutMs) {
    DDebugAssert(mShmImpl != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);
    if (!*mShmImpl->hasValue) {
        auto absTime = boost::posix_time::microsec_clock::universal_time() + boost::posix_time::milliseconds(timeoutMs);
        if (!mShmImpl->cond->timed_wait(lock, absTime, [&] { return *mShmImpl->hasValue; })) {
            return false;
        }
    }
    DDebugAssert(*mShmImpl->hasValue);
    mValueConsumed = true;
    return true;
}

bool FileVoidFuture::IsReady() const {
    // Non-blocking check: the value is ready if hasValue is set and not yet consumed.
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);
    return *mShmImpl->hasValue && !mValueConsumed;
}

// ============================================================
// FileVoidPromise
// ============================================================

FileVoidPromise::FileVoidPromise()
    : VoidPromise(), mShmFileName(), mIsCreator(false), mFutureTaken(false), mShmImpl(nullptr) {
    CreateSharedMemory();
}

FileVoidPromise::~FileVoidPromise() {
    // Note: shared memory file is intentionally NOT removed here.
    // The FileVoidFuture holds a persistent open connection and will
    // clean up when it is destroyed and the last reference is released.
    // In cross-process scenarios, this leaves a small temporary file
    // which is acceptable (persisted in /dev/shm which is tmpfs).
}

void FileVoidPromise::CreateSharedMemory() {
    mShmFileName = GenerateShmFileName();
    mIsCreator = true;
    mShmImpl = std::make_shared<VoidPromiseFutureShmImpl>();
    mShmImpl->Create(mShmFileName);
}

void FileVoidPromise::OpenSharedMemory() {
    mIsCreator = false;
    mShmImpl = std::make_shared<VoidPromiseFutureShmImpl>();
    mShmImpl->Open(mShmFileName);
}

void FileVoidPromise::SetValue() {
    DDebugAssert(mShmImpl != nullptr);
    DDebugAssert(mShmImpl->mutex != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mShmImpl->mutex);

    DDebugAssert(!(*mShmImpl->hasValue) && "SetValue should only be called once");

    *mShmImpl->hasValue = true;

    lock.unlock();
    mShmImpl->cond->notify_all();
}

std::unique_ptr<VoidFuture> FileVoidPromise::GetFuture() {
    if (mFutureTaken) {
        DLogFatal() << "FileVoidPromise::GetFuture() can only be called once.";
        return nullptr;
    }
    mFutureTaken = true;
    return std::make_unique<FileVoidFuture>(mShmFileName);
}

std::string FileVoidPromise::Serialize() const { return mShmFileName; }

void FileVoidPromise::Deserialize(const std::string& data) {
    mShmFileName = data;
    OpenSharedMemory();
}

}  // namespace communication
}  // namespace core
}  // namespace dtorch
