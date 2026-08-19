#!/usr/bin/env bash
#
# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

# Build both the cn and en MkDocs sites and merge them into site/ (matching the GitHub Pages deploy artifact):
#   mkdocs.zh-CN.yml builds docs/cn → site/cn
#   mkdocs.yml       builds docs/en → site/en
#   docs/index.html (language redirect page) → site/index.html
#
# Usage:
#   script/build_docs.sh            # after building, preview the merged site at http://localhost:8000

set -e

DTORCH_ROOT="$(cd "$(dirname "$0")"/..;pwd -P)"
cd "${DTORCH_ROOT}"

SITE_DIR="${DTORCH_ROOT}/site"
BUILD_DIR="${DTORCH_ROOT}/build"

rm -rf "${SITE_DIR}" "${BUILD_DIR}"

mkdocs build --strict -f mkdocs.zh-CN.yml -d build/cn
mkdocs build --strict -f mkdocs.yml -d build/en

mkdir -p "${SITE_DIR}"
cp -r build/cn site/cn
cp -r build/en site/en
cp docs/index.html site/index.html

echo "Docs built to ${SITE_DIR}: index.html (root) + cn/ + en/"

echo "Preview at http://localhost:8000 (Ctrl+C to stop)"
python3 -m http.server 8000 -d "${SITE_DIR}"
