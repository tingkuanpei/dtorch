/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <string>

#include "dtorch/common/utilities.h"

namespace dtorch {

class SubProcessSync {
public:
    SubProcessSync();

    // Open or create a shared memory segment.
    //   create=true  → [parent] create the SHM segment
    //   create=false → [child]  open an existing SHM segment
    SubProcessSync(const std::string& shmFileName, bool create);

    ~SubProcessSync();

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(SubProcessSync);

    // Whether the SHM segment was successfully created/opened (both sides)
    DTORCH_FORCEINLINE bool IsCreated() const noexcept { return mImplPtr != nullptr; }

    // Release the shared memory reference (both sides)
    DTORCH_FORCEINLINE void Reset() { mImplPtr = nullptr; }

    // [parent] Write launch arguments to shared memory for the child to read
    void SetLaunchProcessArgument(const std::string& argumentStr);

    // [child] Read launch arguments from shared memory
    std::string GetLaunchProcessArgument();

    // [parent] Block until the child process signals it has started (isRuned flag set)
    void WaitProcessStarted();

    // [child] Notify the parent that this process has started (sets isRuned flag)
    void NotifyProcessStarted();

    // [parent] Notify the child process to exit (sets exitFlag + notifies condition variable)
    void NotifyExit();

    // [child] Block until the parent signals exit (exitFlag set + condition variable notified).
    void WaitForExit();

private:
    struct Impl;
    std::shared_ptr<Impl> mImplPtr;
};

}  // namespace dtorch
