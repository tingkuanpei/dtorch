/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

namespace boost {
namespace serialization {
class access;
}  // namespace serialization
}  // namespace boost

namespace dtorch {
namespace api {
namespace cpp {
using Serialization = boost::serialization::access;
}  // namespace cpp
}  // namespace api
}  // namespace dtorch
