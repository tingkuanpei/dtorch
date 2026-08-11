/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "environment_variable.h"

#include <cstdlib>
#include <cstring>

#include "dtorch/common/logging.h"
#include "dtorch/common/string.h"

namespace dtorch {

std::unique_ptr<char[]> GetEnv(const std::string& envVar) {
    char* pValue = nullptr;
    size_t len = 0;

#if DTORCH_PLATFORM_WINDOWS
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/dupenv-s-wdupenv-s?view=msvc-170
    if (_dupenv_s(&pValue, &len, envVar.c_str())) {
        return nullptr;
    }
    if (pValue == nullptr) {
        return nullptr;
    }
#else
    // https://zh.cppreference.com/w/cpp/utility/program/getenv
    pValue = std::getenv(envVar.c_str());
    if (pValue == nullptr) {
        return nullptr;
    }
    len = std::strlen(pValue) + 1;
#endif  // _MSC_VER

    auto p = std::make_unique<char[]>(len);
    std::memcpy(p.get(), pValue, len);

#if DTORCH_PLATFORM_WINDOWS
    free(pValue);
#endif

    return p;
}

bool GetEnvVarAsBool(const std::string& envVar, bool defaultValue) {
    bool result = defaultValue;
    auto ptr = GetEnv(envVar);
    if (ptr == nullptr) {
        return result;
    }
    std::string varStr(ptr.get());
    String::ToLower(varStr);

    if (!varStr.empty()) {
        if (varStr == "1" || varStr == "true") {
            result = true;
        } else if (varStr == "0" || varStr == "false") {
            result = false;
        } else {
            DLogError() << "Unacceptable environment variable: " << envVar << " value: " << std::string(ptr.get());
        }
    }

    return result;
}

int GetEnvVarAsInt(const std::string& envVar, int defaultValue) {
    int result;

    auto ptr = GetEnv(envVar);
    if (ptr == nullptr) {
        return defaultValue;
    }
    try {
        result = std::stoi(ptr.get());
    } catch (const std::exception&) {
        DLogError() << "Unacceptable environment variable: " << envVar << " value: " << std::string(ptr.get());
        return defaultValue;
    }
    return result;
}

size_t GetEnvVarAsUInt(const std::string& envVar, size_t defaultValue) {
    auto ptr = GetEnv(envVar);
    if (ptr == nullptr) {
        return defaultValue;
    }

    int result;
    try {
        result = std::stoi(ptr.get());
    } catch (const std::exception&) {
        DLogError() << "Unacceptable environment variable: " << envVar << " value: " << std::string(ptr.get());
        return defaultValue;
    }

    if (result < 0) {
        return defaultValue;
    } else {
        return static_cast<size_t>(result);
    }
}

std::string GetEnvVarAsString(const std::string& envVar, const std::string& defaultValue) {
    auto ptr = GetEnv(envVar);
    if (ptr == nullptr) {
        return defaultValue;
    }
    std::string varStr(ptr.get());
    if (varStr.empty()) {
        return defaultValue;
    }

    return varStr;
}

}  // namespace dtorch
