/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"

namespace dtorch {

class ArgumentParser {
public:
    DTORCH_FORCEINLINE static ArgumentParser& GetSingleton() {
        static ArgumentParser parser;
        return parser;
    }

public:
    void Init(int argc, char** argv);

    void Init(const std::vector<std::string>& arguments);

    DTORCH_FORCEINLINE bool IsInit() const noexcept { return !mProgramName.empty(); }

    DTORCH_FORCEINLINE std::string ProgramName() const noexcept { return mProgramName; }

    bool HasOption(std::string option) const noexcept {
        NormalizeKey(option);
        return mArgv.find(option) != mArgv.end();
    }

    DTORCH_FORCEINLINE std::string OptionValue(std::string option) const {
        NormalizeKey(option);
        DDebugAssert(mArgv.find(option) != mArgv.end());
        return mArgv.at(option);
    }

    std::string ToString() const;

    DTORCH_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const ArgumentParser& parser) {
        os << parser.ToString();
        return os;
    }

private:
    ArgumentParser() : mProgramName(), mArgv() {}

    void AddArgument(const std::string& argument);

    static void NormalizeKey(std::string& key) {
        for (char& c : key) {
            if (c == '-') c = '_';
        }
    }

private:
    std::string mProgramName;
    std::unordered_map<std::string, std::string> mArgv;
};

}  // namespace dtorch
