#!/bin/bash
# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)
#
# Launcher for DTorch CLI tools.
#   dtorch_launcher --remote-runner [args...]  -> runs start_remote_runner_in_process_main
#   dtorch_launcher [args...]                   -> runs start_worker_node_main (default)
# Sets up LD_LIBRARY_PATH for libtorch.so if found.

set -e

# Find libtorch.so directory without importing torch
TORCH_LIB=$(python3 -c "
import os, site, glob
for sp in site.getsitepackages():
    torch_dir = os.path.join(sp, 'torch')
    if not os.path.isdir(torch_dir):
        continue
    for lib in ('lib', 'lib64'):
        d = os.path.join(torch_dir, lib)
        if os.path.isdir(d) and glob.glob(os.path.join(d, 'libtorch*.so')):
            print(d)
            raise SystemExit(0)
" 2>/dev/null)

# Select binary based on first argument
TOOL_NAME="start_worker_node_main"
if [ "$1" = "--remote-runner" ]; then
    TOOL_NAME="start_remote_runner_in_process_main"
    shift
fi

# Find the DTorch tool binary
EXEC_PATH=$(python3 -c "
import os, site
name = 'dtorch/tools/${TOOL_NAME}'
for sp in site.getsitepackages():
    p = os.path.join(sp, name)
    if os.path.isfile(p):
        print(p)
        break
" 2>/dev/null)

if [ -z "$EXEC_PATH" ] || [ ! -f "$EXEC_PATH" ]; then
    echo "Error: Cannot find $TOOL_NAME. Is dtorch installed?" >&2
    exit 1
fi

if [ -z "$TORCH_LIB" ]; then
    echo "Error: Cannot find libtorch.so. Is torch installed?" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH}"

exec "$EXEC_PATH" "$@"
