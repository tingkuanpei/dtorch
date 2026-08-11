/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "string.h"

#include <algorithm>
#include <cctype>

namespace dtorch {

std::vector<std::string> String::Split(const std::string& str, const std::string& delimiters) {
    std::vector<std::string> result;

    std::string::size_type lastPos = 0;
    std::string::size_type pos = str.find_first_of(delimiters);

    while (pos != std::string::npos) {
        result.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = pos + 1;
        pos = str.find_first_of(delimiters, lastPos);
    }
    result.push_back(str.substr(lastPos, pos - lastPos));

    return result;
}

void String::ToUpper(std::string& str) {
    std::for_each(str.begin(), str.end(), [](char& c) { c = static_cast<char>(std::toupper(c)); });
}

void String::ToLower(std::string& str) {
    std::for_each(str.begin(), str.end(), [](char& c) { c = static_cast<char>(std::tolower(c)); });
}

bool String::FindStringBetweenLeftAndRight(const std::string& str, const std::string& leftStr,
                                           const std::string& rightStr, std::string& result) {
    auto leftPos = str.find(leftStr);
    if (leftPos == std::string::npos) {
        result = str;
        return false;
    }
    leftPos = leftPos + leftStr.size();

    auto rightPos = str.substr(leftPos).rfind(rightStr);
    if (rightPos == std::string::npos) {
        result = str;
        return false;
    }
    rightPos = rightPos + leftPos;

    result = str.substr(leftPos, rightPos - leftPos);
    return true;
}

}  // namespace dtorch
