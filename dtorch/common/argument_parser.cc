/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "argument_parser.h"

#include <sstream>
#include <stdexcept>

#include "dtorch/common/debug.h"
#include "dtorch/common/logging.h"
#include "dtorch/common/string.h"

namespace dtorch {

void ArgumentParser::Init(const std::vector<std::string>& arguments) {
    DAlwaysAssert(!arguments.empty());
    mArgv.clear();
    mProgramName = arguments[0];

    for (size_t i = 1; i < arguments.size(); ++i) {
        AddArgument(arguments[i]);
    }
}

void ArgumentParser::Init(int argc, char** argv) {
    DAlwaysAssert(argc > 0);
    mArgv.clear();
    mProgramName = argv[0];

    for (int i = 1; i < argc; i++) {
        AddArgument(argv[i]);
    }
}

std::string ArgumentParser::ToString() const {
    std::stringstream ss;
    ss << "ProgramName: " << mProgramName << ", Argument: [";
    for (auto it : mArgv) {
        ss << "{" << it.first << ", " << it.second << "}, ";
    }
    ss << "]";
    return ss.str();
}

void ArgumentParser::AddArgument(const std::string& argument) {
    std::string input = argument;
    // erase --
    if (input.size() > 2 && input[0] == '-' && input[1] == '-') {
        input = input.substr(2);
    }

    std::vector<std::string> splitStr = String::Split(input, "=");
    if (splitStr.size() == 0 || splitStr.size() > 2) {
        throw std::invalid_argument("Invalid argument format: " + argument);
    }

    NormalizeKey(splitStr[0]);

    if (mArgv.count(splitStr[0]) > 0) {
        throw std::invalid_argument("Duplicate argument: " + splitStr[0]);
    }

    if (splitStr.size() >= 2) {
        mArgv[splitStr[0]] = splitStr[1];
    } else {
        mArgv[splitStr[0]] = "";
    }
}

}  // namespace dtorch
