# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Develop

@docs/cn/get_started/how_to_build.md

## Testing

@docs/cn/get_started/test.md

## Code Formatting
```bash
pre-commit run -a          # Run all checks (clang-format + black)
```
C++: Google style, clang-format, ColumnLimit=120, IndentWidth=4. Use camelCase naming.
Python: Black with `--line-length=120`.

## Developer Guide

@docs/cn/developer_guide/project_overview.md

@docs/cn/developer_guide/document_index.md
