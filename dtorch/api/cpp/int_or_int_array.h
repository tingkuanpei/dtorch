/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <sstream>
#include <vector>

#include "api_utilities.h"

namespace dtorch {
namespace api {
namespace cpp {

class IntOrIntArray : public std::vector<int64_t> {
public:
    IntOrIntArray() : std::vector<int64_t>() {}

    IntOrIntArray(int64_t oneElt) : std::vector<int64_t>({oneElt}) {}

    IntOrIntArray(const std::vector<int64_t>& vec) : std::vector<int64_t>(vec.begin(), vec.end()) {}

    std::vector<int64_t> Vec() const { return std::vector<int64_t>(begin(), end()); }

    DTORCH_API_FORCEINLINE std::string ToString() const {
        std::stringstream ss;
        ss << "(";
        if (this->size() > 0) {
            ss << (*this)[0];
        }
        for (size_t i = 1; i < this->size(); i++) {
            ss << ", " << (*this)[i];
        }
        ss << ")";
        return ss.str();
    }

    DTORCH_API_FORCEINLINE friend std::ostream& operator<<(std::ostream& os, const IntOrIntArray& vec) {
        os << vec.ToString();
        return os;
    }
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
