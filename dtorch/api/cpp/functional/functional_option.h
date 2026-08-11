/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>
#include <string>

#include "../serialization.h"

namespace dtorch {
namespace api {
namespace cpp {
namespace functional {

struct SdpaOption {
    // candidate: "", "auto", "qk_int8_pv_fp8", "qk_int8_pv_fp16"
    std::string sageAttentionType;

    SdpaOption() : sageAttentionType("") {}

    friend Serialization;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /*version*/) {
        ar & sageAttentionType;
    }
};

}  // namespace functional
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
