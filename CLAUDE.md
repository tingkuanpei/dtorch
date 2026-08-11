# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Develop

@docs/get_started/how_to_build.md

## Testing

@docs/get_started/test.md

## Code formatting
```bash
pre-commit run -a          # Run all checks (clang-format + black)
```
C++: Google style, clang-format, ColumnLimit=120, IndentWidth=4. Use camelCase naming.
Python: Black with `--line-length=120`.

## Architecture

@docs/developer_guide/architecture.md
