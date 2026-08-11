/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "sub_process_sync.h"

#include "dtorch/common/debug.h"
#include "dtorch/external/boost/boost_interprocess.h"

namespace dtorch {

struct SubProcessSync::Impl {
    std::shared_ptr<external::boost::ShmAutoRemove> remover;
    std::shared_ptr<external::boost::ManagedSharedMemory> memory;
    bool* isRuned;
    bool* exitFlag;
    external::boost::InterprocessMutex* mutex;
    external::boost::InterprocessCondition* cv;

public:
    Impl(const std::string& shmFileName, bool create)
        : remover(), memory(), isRuned(nullptr), exitFlag(nullptr), mutex(nullptr), cv(nullptr) {
        if (create) {
            remover = std::make_shared<external::boost::ShmAutoRemove>(shmFileName);
            memory = std::make_shared<external::boost::ManagedSharedMemory>(::boost::interprocess::create_only,
                                                                            shmFileName, 4096);
            isRuned = memory->Construct<bool>("IsRuned", false);
            exitFlag = memory->Construct<bool>("ExitFlag", false);
            mutex = memory->Construct<external::boost::InterprocessMutex>("Mutex");
            cv = memory->Construct<external::boost::InterprocessCondition>("Cond");
        } else {
            memory =
                std::make_shared<external::boost::ManagedSharedMemory>(::boost::interprocess::open_only, shmFileName);
            isRuned = memory->Find<bool>("IsRuned");
            exitFlag = memory->Find<bool>("ExitFlag");
            mutex = memory->Find<external::boost::InterprocessMutex>("Mutex");
            cv = memory->Find<external::boost::InterprocessCondition>("Cond");
        }
    }

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(Impl);
};

SubProcessSync::SubProcessSync() : mImplPtr() {}

SubProcessSync::SubProcessSync(const std::string& shmFileName, bool create)
    : mImplPtr(std::make_shared<Impl>(shmFileName, create)) {}

SubProcessSync::~SubProcessSync() = default;

void SubProcessSync::SetLaunchProcessArgument(const std::string& argumentStr) {
    DDebugAssert(mImplPtr != nullptr);
    mImplPtr->memory->ConstructString("LaunchProcessArgument", argumentStr);
}

std::string SubProcessSync::GetLaunchProcessArgument() { return mImplPtr->memory->FindStr("LaunchProcessArgument"); }

void SubProcessSync::WaitProcessStarted() {
    DDebugAssert(mImplPtr != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mImplPtr->mutex);
    mImplPtr->cv->wait(lock, [=]() { return *mImplPtr->isRuned; });
}

void SubProcessSync::NotifyProcessStarted() {
    DDebugAssert(mImplPtr != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mImplPtr->mutex);
    *mImplPtr->isRuned = true;
    mImplPtr->cv->notify_all();
}

void SubProcessSync::NotifyExit() {
    DDebugAssert(mImplPtr != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mImplPtr->mutex);
    *mImplPtr->exitFlag = true;
    mImplPtr->cv->notify_all();
}

void SubProcessSync::WaitForExit() {
    DDebugAssert(mImplPtr != nullptr);
    external::boost::ScopedLock<external::boost::InterprocessMutex> lock(*mImplPtr->mutex);
    if (!*mImplPtr->exitFlag) {
        mImplPtr->cv->wait(lock, [this] { return *mImplPtr->exitFlag; });
    }
}

}  // namespace dtorch
