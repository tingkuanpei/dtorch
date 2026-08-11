/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#pragma once

#include <torch/torch.h>

#include "dtorch/api/cpp/cluster.h"
#include "nanobind_register.h"

namespace dtorch {
namespace api {
namespace py {

class PyBindCluster {
public:
    static std::string ModuleName() { return "cluster"; }

    static void RegisterFunc(nb::module_& m) {
        using api::cpp::distributed::MainNode;
        m.def("_main_node_set_address", [](const std::string& address) { MainNode::SetMainNodeAddress(address); });

        m.def("_main_node_wait_cluster_ready", [](size_t numNodes, double timeoutSecond) {
            return MainNode::WaitClusterReady(numNodes, timeoutSecond);
        });

        m.def("_main_node_get_address", []() { return MainNode::GetMainNodeAddress(); });

        m.def("_main_node_num_node_in_cluster", []() { return MainNode::NumNodeInCluster(); });
    }
};

}  // namespace py
}  // namespace api
}  // namespace dtorch
