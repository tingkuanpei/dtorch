#!/usr/bin/env bash
#
# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

# Reference:
#     https://github.com/grpc/grpc/blob/v1.76.0/BUILDING.md
#     https://github.com/grpc/grpc/blob/v1.76.0/test/distrib/cpp/run_distrib_test_cmake_module_install.sh

DTORCH_ROOT="$(cd "$(dirname "$0")"/..;pwd -P)"
GRPC_BUILD_ROOT="$DTORCH_ROOT/third_party_install"

apt-get update && apt-get install -y libssl-dev
apt-get install -y build-essential autoconf libtool pkg-config

mkdir -p $GRPC_BUILD_ROOT
pushd $GRPC_BUILD_ROOT
GRPC_TAR_FILE=$DTORCH_ROOT/third_party_local_url/grpc-1.76.0.tar.gz
if [ -e "$GRPC_TAR_FILE" ]; then
    tar -xf $GRPC_TAR_FILE
else
    git clone --recurse-submodules -b v1.76.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc
fi

mkdir -p $GRPC_BUILD_ROOT/grpc/cmake/build
pushd $GRPC_BUILD_ROOT/grpc/cmake/build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$GRPC_BUILD_ROOT/grpc_install \
    -DCMAKE_CXX_STANDARD=17 \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_SSL_PROVIDER=package \
    ../..
make -j12 install

popd
popd
