/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <filesystem>
#include <string>

#include "dtorch/common/utilities.h"

namespace dtorch {

std::string GetRandomFileName(size_t length = 12);

std::string GetShmDirectoryPath();

std::string GetTempDirectoryPath();

std::string GetTmpFilePath();

class FileRemoveGuard {
public:
    FileRemoveGuard(const std::string& filePath) : mFilePath(std::filesystem::absolute(filePath)) {}

    ~FileRemoveGuard();

    DTORCH_DISABLE_COPY_AND_DEFAULT_MOVE(FileRemoveGuard);

    bool IsFileExist() const noexcept { return std::filesystem::exists(mFilePath); }

    bool IsParentPathExist() const noexcept {
        std::filesystem::path parentPath = mFilePath.parent_path();
        return std::filesystem::is_directory(parentPath) && std::filesystem::exists(parentPath);
    }

private:
    std::filesystem::path mFilePath;
};

}  // namespace dtorch
