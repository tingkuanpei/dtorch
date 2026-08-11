/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "dtorch/api/cpp/cluster.h"

using dtorch::api::cpp::distributed::WorkerNode;

int main(int argc, char** argv) {
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; i++) {
        arguments.push_back(argv[i]);
    }

    return WorkerNode::ExecMain(arguments);
}
