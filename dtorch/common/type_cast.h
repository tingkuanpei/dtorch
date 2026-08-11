/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <memory>
#include <type_traits>

#include "dtorch/common/logging.h"
#include "dtorch/common/utilities.h"

namespace dtorch {

template <typename DerivedClass, typename BaseClass>
DTORCH_FORCEINLINE DerivedClass* DerivedCast(BaseClass* p) {
    static_assert(std::is_base_of<BaseClass, DerivedClass>::value, "DerivedClass MUST derived from BaseClass");

    DerivedClass* result = dynamic_cast<DerivedClass*>(p);
    if (!result) {
        DLogFatal() << "Dynamic_cast from base to derived failed";
        return nullptr;
    }
    return result;
}

template <typename DerivedClass, typename BaseClass>
DTORCH_FORCEINLINE const DerivedClass* DerivedCast(const BaseClass* p) {
    static_assert(std::is_base_of<BaseClass, DerivedClass>::value, "DerivedClass MUST derived from BaseClass");

    const DerivedClass* result = dynamic_cast<const DerivedClass*>(p);
    if (!result) {
        DLogFatal() << "Dynamic_cast from base to derived failed";
        return nullptr;
    }
    return result;
}

template <typename DerivedClass, typename BaseClass>
DTORCH_FORCEINLINE std::unique_ptr<DerivedClass> DynamicPointerCast(std::unique_ptr<BaseClass>&& r) noexcept {
    static_assert(std::is_base_of<BaseClass, DerivedClass>::value, "DerivedClass MUST derived from BaseClass");
    static_assert(std::has_virtual_destructor<DerivedClass>::value,
                  "The target of dynamic_pointer_cast must have a virtual destructor.");

    DerivedClass* p = dynamic_cast<DerivedClass*>(r.get());
    if (p) {
        r.release();
    } else {
        DLogFatal() << "Dynamic_cast from base to derived failed";
        return nullptr;
    }
    return std::unique_ptr<DerivedClass>(p);
}

template <typename BaseClass, typename DerivedClass>
DTORCH_FORCEINLINE BaseClass& BaseObject(DerivedClass& t) {
    static_assert(std::is_base_of<BaseClass, DerivedClass>::value, "DerivedClass MUST derived from BaseClass");
    return static_cast<BaseClass&>(t);
}

}  // namespace dtorch
