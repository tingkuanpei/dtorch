#!/usr/bin/env bash
#
# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

DTORCH_ROOT="$(cd "$(dirname "$0")"/..;pwd -P)"
THIRD_PARTY_DIR="${DTORCH_ROOT}/third_party_local_url"

mkdir -p $THIRD_PARTY_DIR
cd ${THIRD_PARTY_DIR}
wget -O glog-0.5.0.tar.gz https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz
wget -O googletest-release-1.11.0.tar.gz https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
wget -O concurrentqueue-1.0.3.tar.gz https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.3.tar.gz
wget -O readerwriterqueue-1.0.6.tar.gz https://github.com/cameron314/readerwriterqueue/archive/refs/tags/v1.0.6.tar.gz
git clone --recurse-submodules -b v1.76.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc
tar czf grpc-1.76.0.tar.gz grpc
wget -O cppzmq-4.11.0.tar.gz https://codeload.github.com/zeromq/cppzmq/tar.gz/refs/tags/v4.11.0
wget -O zeromq-4.3.5.tar.gz https://release-assets.githubusercontent.com/github-production-release-asset/263379/6980bffa-a62a-4f09-b37a-26c3c5c475bf?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-03-24T08%3A19%3A54Z&rscd=attachment%3B+filename%3Dzeromq-4.3.5.tar.gz&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-03-24T07%3A19%3A39Z&ske=2026-03-24T08%3A19%3A54Z&sks=b&skv=2018-11-09&sig=7bzZJaS6BYlK%2FFl9qsF0j2N2j4qqlPrkAoZKUfxmpKA%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc3NDMzODAwOSwibmJmIjoxNzc0MzM3NzA5LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.AdoosHmSAg3lT5iDwMRolMCi8okdI6IRI1mC53gcy0Y&response-content-disposition=attachment%3B%20filename%3Dzeromq-4.3.5.tar.gz&response-content-type=application%2Foctet-stream
