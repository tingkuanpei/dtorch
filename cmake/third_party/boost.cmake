# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

# Boost.Serialization required compiled binaries:
# https://www.boost.org/doc/user-guide/header-organization-compilation.html
find_package(Boost QUIET COMPONENTS serialization)

if(NOT Boost_FOUND OR NOT Boost_SERIALIZATION_FOUND)
    message(FATAL_ERROR "Boost not found. Install it first. On Ubuntu: 'apt install libboost-all-dev'.")
endif()

set(BOOST_INCLUDE_DIR       ${Boost_INCLUDE_DIRS})
set(BOOST_LIBRARYS          Boost::serialization)
