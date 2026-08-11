/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace dtorch {

class String {
public:
    static std::vector<std::string> Split(const std::string& str, const std::string& delimiters = " ");

    static void ToUpper(std::string& str);

    static void ToLower(std::string& str);

    static bool FindStringBetweenLeftAndRight(const std::string& str, const std::string& leftStr,
                                              const std::string& rightStr, std::string& result);

    template <typename T>
    static std::string ToString(const std::vector<T>& vec);

    template <typename T>
    static std::string ToString(const std::unordered_set<T>& vec);
};

template <typename T>
std::string String::ToString(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i];
        if (i != vec.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

template <typename T>
std::string String::ToString(const std::unordered_set<T>& set) {
    std::stringstream ss;
    ss << "[";
    for (const auto& it : set) {
        ss << it << ", ";
    }
    ss << "]";
    return ss.str();
}

}  // namespace dtorch
