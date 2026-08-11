/*
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
*/

#include "dtorch/api/cpp/dtorch.h"
#include "dtorch/common/argument_parser.h"
#include "dtorch/common/logging.h"
#include "test.h"

// --gtest_filter=SpecificTest.*
int main(int argc, char** argv) {
    auto& parser = dtorch::ArgumentParser::GetSingleton();
    parser.Init(argc, argv);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
