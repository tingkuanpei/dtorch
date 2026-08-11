/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <cstdint>
#include <type_traits>

#include "serialization.h"

namespace dtorch {
namespace api {
namespace cpp {

class Scalar {
public:
    Scalar() : Scalar(int32_t(0)) {}

    template <typename T,
              typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, int>::type = 0>
    Scalar(const T& value) : mValue(), mActiveTag(HAS_S) {
        mValue.s = value;
    }

    template <typename T,
              typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, int>::type = 0>
    Scalar(const T& value) : mValue(), mActiveTag(HAS_U) {
        mValue.u = value;
    }

    template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
    Scalar(const T& value) : mValue(), mActiveTag(HAS_D) {
        mValue.d = value;
    }

    Scalar(const Scalar& other) : mValue(other.mValue), mActiveTag(other.mActiveTag) {}

    Scalar& operator=(const Scalar& other) {
        mValue = other.mValue;
        mActiveTag = other.mActiveTag;
        return *this;
    }

    template <typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    T Value() const {
        switch (mActiveTag) {
            case HAS_S:
                return static_cast<T>(mValue.s);
            case HAS_U:
                return static_cast<T>(mValue.u);
            case HAS_D:
                return static_cast<T>(mValue.d);
            default:
                return static_cast<T>(0);
        }
    }

    bool IsIntegral() const { return mActiveTag == HAS_S || mActiveTag == HAS_U; }
    bool IsFloatingPoint() const { return mActiveTag == HAS_D; }
    bool IsSigned() const { return mActiveTag == HAS_S || mActiveTag == HAS_D; }
    bool IsUnsigned() const { return mActiveTag == HAS_U; }

    Scalar operator+(const Scalar& other);
    Scalar operator-(const Scalar& other);
    Scalar operator*(const Scalar& other);
    Scalar operator/(const Scalar& other);

    Scalar& operator+=(const Scalar& other);
    Scalar& operator-=(const Scalar& other);
    Scalar& operator*=(const Scalar& other);
    Scalar& operator/=(const Scalar& other);

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & mActiveTag;
        switch (mActiveTag) {
            case HAS_S:
                ar & mValue.s;
                break;
            case HAS_U:
                ar & mValue.u;
                break;
            case HAS_D:
                ar & mValue.d;
                break;
            default:
                break;
        }
    }

private:
    union Value {
        int64_t s;
        uint64_t u;
        double d;
    } mValue;
    enum { HAS_S, HAS_U, HAS_D, HAS_NONE } mActiveTag;
};

}  // namespace cpp
}  // namespace api
}  // namespace dtorch
