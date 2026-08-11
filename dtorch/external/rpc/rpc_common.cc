/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "rpc_common.h"

#include "dtorch/common/debug.h"
#include "dtorch/common/filesystem.h"

namespace dtorch {
namespace external {
namespace rpc {

std::string GetRandomUdsAddress(size_t length) {
    return "unix:" + GetTempDirectoryPath() + "/DTorch_RPC_UDS_" + GetRandomFileName(length) + ".sock";
}

bool IsUdsAddress(const std::string& address) {
    const std::string udsPrefix = "unix:";
    if (address.size() < udsPrefix.size()) {
        return false;
    }
    return address.substr(0, udsPrefix.size()) == udsPrefix;
}

std::string GetUdsActualFilePath(const std::string& address) {
    DDebugAssert(IsUdsAddress(address));
    const std::string udsPrefix = "unix:";
    return address.substr(udsPrefix.size());
}

std::unique_ptr<FileRemoveGuard> GetUdsFileRemoveGuard(const std::string& address) {
    std::unique_ptr<FileRemoveGuard> result;
    if (IsUdsAddress(address)) {
        result = std::make_unique<FileRemoveGuard>(GetUdsActualFilePath(address));
        if (result->IsFileExist()) {
            throw std::runtime_error("UDS address already exist: " + address);
        }
        if (!result->IsParentPathExist()) {
            throw std::runtime_error("UDS address parent path not exist: " + address);
        }
    }
    return result;
}

}  // namespace rpc
}  // namespace external
}  // namespace dtorch
