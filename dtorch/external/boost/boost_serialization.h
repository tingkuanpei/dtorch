/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <optional>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/serialization/vector.hpp>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// Boost.Serialization Doc:
// https://www.boost.org/doc/libs/1_90_0/libs/serialization/doc/index.html
// https://www.boost.org/doc/libs/1_90_0/libs/serialization/doc/tutorial.html#archives

namespace boost {
namespace serialization {

template <typename Archive, typename T>
void serialize(Archive& ar, std::optional<T>& opt, const unsigned int /*version*/) {
    bool hasValue = opt.has_value();
    ar & hasValue;

    if (hasValue) {
        if constexpr (Archive::is_saving::value) {
            ar & opt.value();
        } else {
            T value;
            ar & value;
            opt = value;
        }
    } else if constexpr (Archive::is_loading::value) {
        opt.reset();
    }
}

}  // namespace serialization
}  // namespace boost

namespace dtorch {
namespace external {
namespace boost {

using BinaryOArchive = ::boost::archive::binary_oarchive;
using BinaryIArchive = ::boost::archive::binary_iarchive;

}  // namespace boost
}  // namespace external
}  // namespace dtorch
