/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "boost_interprocess.h"

#include <filesystem>
#include <sstream>
#include <string>

#include "dtorch/common/filesystem.h"
#include "dtorch/common/logging.h"
#include "dtorch/core/communication/global_instance_id.h"

namespace dtorch {
namespace external {
namespace boost {

std::once_flag ManagedSharedMemory::kCleanUpOnceFlag;

std::string ManagedSharedMemory::GetShmFileNamePrefix(bool withInstanceId) {
    std::stringstream ss;
    ss << "DTorchSharedMemoryFile_";
    if (withInstanceId) {
        ss << "Instance_" << core::communication::GlobalCommInstanceId::GetSingleton().GetInstanceId() << "_";
    }
    return ss.str();
}

void ManagedSharedMemory::CleanUpResidualSharedMemoryFiles() {
    const std::string shmFolder = GetShmDirectoryPath();

    // Use std::filesystem::directory_iterator to traverse all files in shmFolder directory, and delete files whose
    // name prefix satisfies kShmFileNamePrefix
    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(shmFolder))) {
        if (entry.is_regular_file() && entry.path().string().find(GetShmFileNamePrefix()) != std::string::npos) {
            std::filesystem::remove(entry.path());
            DLogError() << "Remove residual shared memory file: " << entry.path().string();
        }
    }
}

// Store string as vector
// https://www.boost.org/doc/libs/latest/doc/html/interprocess/quick_guide.html#interprocess.quick_guide.qg_interprocess_container
void ManagedSharedMemory::ConstructString(const std::string& key, const std::string& strContent) {
    const CharAllocator allocInst(mMemory->get_segment_manager());
    try {
        ShmVector* vector = mMemory->construct<ShmVector>(key.c_str())(allocInst);
        vector->assign(strContent.begin(), strContent.end());
    } catch (std::exception& e) {
        DLogFatal() << "Shared memory ConstructString error, msg: " << e.what();
    }
}

std::string ManagedSharedMemory::FindStr(const std::string& key) {
    std::pair<ShmVector*, size_type> value = mMemory->find<ShmVector>(key.c_str());
    DDebugAssert(value.first);
    DDebugAssert(value.second == 1);

    std::string result;
    result.assign(value.first->begin(), value.first->end());
    return result;
}

}  // namespace boost
}  // namespace external
}  // namespace dtorch
