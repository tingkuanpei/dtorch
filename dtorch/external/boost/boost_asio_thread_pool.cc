/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "boost_asio_thread_pool.h"

namespace dtorch {
namespace external {
namespace boost {

BoostAsioThreadPool& BoostAsioThreadPool::GetInstance() {
    static BoostAsioThreadPool instance;
    return instance;
}

BoostAsioThreadPool::BoostAsioThreadPool() : mPool(12) {}

BoostAsioThreadPool::~BoostAsioThreadPool() { mPool.join(); }

}  // namespace boost
}  // namespace external
}  // namespace dtorch
