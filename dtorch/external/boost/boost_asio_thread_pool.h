/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <boost/asio.hpp>

#include "dtorch/common/utilities.h"

namespace dtorch {
namespace external {
namespace boost {

class BoostAsioThreadPool {
public:
    static BoostAsioThreadPool& GetInstance();

    DTORCH_DISABLE_COPY_AND_MOVE(BoostAsioThreadPool);

    // Post a task to the thread pool. F must be copy-constructible.
    template <typename F>
    void Post(F&& task) {
        ::boost::asio::post(mPool, std::forward<F>(task));
    }

private:
    BoostAsioThreadPool();
    ~BoostAsioThreadPool();

    ::boost::asio::thread_pool mPool;
};

}  // namespace boost
}  // namespace external
}  // namespace dtorch
