/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "filesystem.h"

#include <chrono>
#include <random>
#include <sstream>

#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"

namespace dtorch {

std::string GetRandomFileName(size_t length) {
    const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<> dis(0, static_cast<std::mt19937::result_type>(characters.size() - 1));

    std::stringstream ss;
    for (size_t i = 0; i < length; ++i) {
        ss << characters[dis(gen)];
    }
    return ss.str();
}

std::string GetShmDirectoryPath() {
#if DTORCH_PLATFORM_LINUX
    const std::string kShmFolder = "/dev/shm";
#elif DTORCH_PLATFORM_WINDOWS
    DUnimplemented();
#endif
    if (!std::filesystem::exists(kShmFolder) || !std::filesystem::is_directory(kShmFolder)) {
        DLogFatal() << "GetShmDirectoryPath error: " << kShmFolder << " is not exist or not a directory.";
        return "";
    }

    return kShmFolder;
}

std::string GetTempDirectoryPath() {
    try {
        const std::string tempDirectoryPath = std::filesystem::temp_directory_path();
        return tempDirectoryPath;
    } catch (const std::exception& e) {
        DLogFatal() << "GetTempDirectoryPath error, error message: " << e.what()
                    << ". Please check if the temp directory is exist and writable";
        return "";
    }
}

std::string GetTmpFilePath() {
    std::string result = GetTempDirectoryPath() + "/DTorch_" + GetRandomFileName();
    if (std::filesystem::exists(result)) {
        DLogFatal() << "GetTmpFilePath error: " << result << " already exists.";
        return "";
    }
    return result;
}

FileRemoveGuard::~FileRemoveGuard() {
    if (!std::filesystem::exists(mFilePath)) {
        return;
    }

    try {
        if (std::filesystem::is_regular_file(mFilePath)) {
            std::filesystem::remove(mFilePath);
        } else if (std::filesystem::is_directory(mFilePath)) {
            std::filesystem::remove_all(mFilePath);
        }
    } catch (const std::exception& e) {
        DLogFatal() << "FileRemoveGuard destructor error: " << e.what() << " file: " << mFilePath << std::endl;
    }
}

}  // namespace dtorch
