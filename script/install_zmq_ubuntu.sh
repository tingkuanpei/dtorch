#!/usr/bin/env bash
#
# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

DTORCH_ROOT="$(cd "$(dirname "$0")"/..;pwd -P)"
ZMQ_BUILD_ROOT="$DTORCH_ROOT/third_party_install"
mkdir -p $ZMQ_BUILD_ROOT
pushd $ZMQ_BUILD_ROOT

# libzmq
ZMQ_TAR_FILE=$DTORCH_ROOT/third_party_local_url/zeromq-4.3.5.tar.gz
if [ -e "$ZMQ_TAR_FILE" ]; then
    tar -xf $ZMQ_TAR_FILE --transform='s,^zeromq-4.3.5,libzmq,'
else
    git clone -b v4.3.5 --depth 1 --shallow-submodules https://github.com/zeromq/libzmq.git
fi

mkdir -p libzmq/build
pushd libzmq/build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$ZMQ_BUILD_ROOT/libzmq_install \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    ..
make -j12 install

popd

# cppzmq
ZMQ_TAR_FILE=$DTORCH_ROOT/third_party_local_url/cppzmq-4.11.0.tar.gz
if [ -e "$ZMQ_TAR_FILE" ]; then
    tar -xf $ZMQ_TAR_FILE --transform='s,^cppzmq-4.11.0,cppzmq,'
else
    git clone -b v4.11.0 --depth 1 --shallow-submodules https://github.com/zeromq/cppzmq.git
fi
mkdir -p cppzmq/build
pushd cppzmq/build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCPPZMQ_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=$ZMQ_BUILD_ROOT/cppzmq_install \
    -DCMAKE_PREFIX_PATH=$ZMQ_BUILD_ROOT/libzmq_install \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    ..
make -j12 install

popd
popd
