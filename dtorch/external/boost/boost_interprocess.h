/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <boost/container/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/set.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"
#include "dtorch/core/type.h"

namespace dtorch {
namespace external {
namespace boost {

// Basic types

using ManagedSharedMemorySegmentManager = ::boost::interprocess::managed_shared_memory::segment_manager;
template <class T, class SegmentManager>
using Allocator = ::boost::interprocess::allocator<T, SegmentManager>;
using CharAllocator = Allocator<char, ManagedSharedMemorySegmentManager>;

using ShmString = ::boost::container::basic_string<char, std::char_traits<char>, CharAllocator>;
using ShmStringAllocator = Allocator<ShmString, ManagedSharedMemorySegmentManager>;
using ShmStringSet = ::boost::container::set<ShmString, std::less<ShmString>, ShmStringAllocator>;

// map 的 value_type 是 pair<const Key, T>，分配器必须与之一致
using ShmStringIntAllocator = Allocator<std::pair<const ShmString, int64_t>, ManagedSharedMemorySegmentManager>;
using ShmStringIntMap = ::boost::container::map<ShmString, int64_t, std::less<ShmString>, ShmStringIntAllocator>;

using ShmStringDeviceKindAllocator =
    Allocator<std::pair<const ShmString, core::DeviceKind>, ManagedSharedMemorySegmentManager>;
using ShmStringDeviceKindMap =
    ::boost::container::map<ShmString, core::DeviceKind, std::less<ShmString>, ShmStringDeviceKindAllocator>;

using ShmVector = ::boost::container::vector<char, CharAllocator>;

using InterprocessMutex = ::boost::interprocess::interprocess_mutex;
using InterprocessCondition = ::boost::interprocess::interprocess_condition;

template <typename MutexType>
using ScopedLock = ::boost::interprocess::scoped_lock<MutexType>;

// ShmAutoRemove

struct ShmAutoRemove {
    ShmAutoRemove(const std::string& name) : mName(name) {
        ::boost::interprocess::shared_memory_object::remove(mName.c_str());
    }
    ~ShmAutoRemove() { ::boost::interprocess::shared_memory_object::remove(mName.c_str()); }

private:
    std::string mName;
};

// ManagedSharedMemory

class ManagedSharedMemory {
public:
    static std::once_flag kCleanUpOnceFlag;

    static DTORCH_FORCEINLINE std::string GetShmFileNameWithPrefix(const std::string& name) {
        return GetShmFileNamePrefix() + name;
    }

    static std::string GetShmFileNamePrefix(bool withInstanceId = true);

    static void CleanUpResidualSharedMemoryFiles();

public:
    using size_type = ::boost::interprocess::managed_shared_memory::size_type;

    template <typename ModeType, typename... Args>
    ManagedSharedMemory(ModeType mode, const std::string& name, Args&&... args) : mMemory() {
        std::call_once(ManagedSharedMemory::kCleanUpOnceFlag, ManagedSharedMemory::CleanUpResidualSharedMemoryFiles);

        if (name.find(GetShmFileNamePrefix(false)) == std::string::npos) {
            DLogFatal() << "Shared memory name must contains prefix: " << GetShmFileNamePrefix(false)
                        << " but got: " << name
                        << ". Please use ManagedSharedMemory::GetShmFileNameWithPrefix() to get the name with prefix.";
        }

        try {
            mMemory = std::make_unique<::boost::interprocess::managed_shared_memory>(mode, name.c_str(), args...);
        } catch (std::exception& e) {
            std::stringstream ss;
            ss << "Open shared memory failed, file name: " << name << ", error msg: " << e.what();
            DLogFatal() << ss.str();
        }
    }

    template <class T, typename... Args>
    T* FindOrConstruct(const std::string& key, Args&&... args) {
        try {
            return mMemory->find_or_construct<T>(key.c_str())(args...);
        } catch (std::exception& e) {
            DLogFatal() << "Shared memory FindOrConstruct error, msg: " << e.what();
        }
        return nullptr;
    }

    template <class T, typename... Args>
    T* Construct(const std::string& key, Args&&... args) {
        try {
            DAlwaysAssert(Count<T>(key) == 0);
            return mMemory->construct<T>(key.c_str())(args...);
        } catch (std::exception& e) {
            DLogFatal() << "Shared memory Construct error, msg: " << e.what();
        }
        return nullptr;
    }

    template <class T, typename... Args>
    T* ConstructArray(const std::string& key, const size_type count, Args&&... args) {
        try {
            DAlwaysAssert(Count<T>(key) == 0);
            return mMemory->construct<T>(key.c_str())[count](args...);
        } catch (std::exception& e) {
            DLogFatal() << "Shared memory ConstructArray error, msg: " << e.what();
        }
        return nullptr;
    }

    void ConstructString(const std::string& key, const std::string& strContent);

    ShmStringSet* FindOrConstructStringSet(const std::string& key, const ShmStringAllocator& allocInst) {
        try {
            return mMemory->find_or_construct<ShmStringSet>(key.c_str())(std::less<ShmString>(), allocInst);
        } catch (std::exception& e) {
            DLogFatal() << "Shared memory FindOrConstructStringSet error, msg: " << e.what();
        }
        return nullptr;
    }

    ShmStringIntMap* FindOrConstructStringIntMap(const std::string& key, const ShmStringIntAllocator& allocInst) {
        try {
            return mMemory->find_or_construct<ShmStringIntMap>(key.c_str())(std::less<ShmString>(), allocInst);
        } catch (std::exception& e) {
            DLogFatal() << "Shared memory FindOrConstructStringIntMap error, msg: " << e.what();
        }
        return nullptr;
    }

    ShmStringDeviceKindMap* FindOrConstructStringDeviceKindMap(const std::string& key,
                                                               const ShmStringDeviceKindAllocator& allocInst) {
        try {
            return mMemory->find_or_construct<ShmStringDeviceKindMap>(key.c_str())(std::less<ShmString>(), allocInst);
        } catch (std::exception& e) {
            DLogFatal() << "Shared memory FindOrConstructStringDeviceKindMap error, msg: " << e.what();
        }
        return nullptr;
    }

    ShmStringSet* ConstructStringSet(const std::string& key, const ShmStringAllocator& allocInst) {
        try {
            DAlwaysAssert(Count<ShmStringSet>(key) == 0);
            return mMemory->construct<ShmStringSet>(key.c_str())(std::less<ShmString>(), allocInst);
        } catch (std::exception& e) {
            DLogFatal() << "Shared memory ConstructStringSet error, msg: " << e.what();
        }
        return nullptr;
    }

    ShmStringSet* FindStringSet(const std::string& key) {
        std::pair<ShmStringSet*, size_type> value = mMemory->find<ShmStringSet>(key.c_str());
        DDebugAssert(value.first);
        DDebugAssert(value.second == 1);
        return value.first;
    }

    template <class T>
    T* Find(const std::string& key, const size_type count = 1) {
        std::pair<T*, size_type> value = mMemory->find<T>(key.c_str());
        DDebugAssertMsg(value.first, "Get named shared memory objects failed, name: " + key);
        DDebugAssert(value.second == count);
        return value.first;
    }

    std::string FindStr(const std::string& key);

    template <class T>
    size_t Count(const std::string& key) {
        std::pair<T*, size_type> value = mMemory->find<T>(key.c_str());
        return value.first == nullptr ? 0 : 1;
    }

    template <class T>
    void Destroy(const std::string& key) {
        DDebugAssert(mMemory->destroy<T>(key.c_str()));
    }

public:
    std::unique_ptr<::boost::interprocess::managed_shared_memory> mMemory;
};

}  // namespace boost
}  // namespace external
}  // namespace dtorch
