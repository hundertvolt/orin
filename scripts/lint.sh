#!/usr/bin/env bash
# Runs the same lint check as the CI "ruff" job. Lint only - ruff format is
# deliberately not part of this toolchain, see pyproject.toml.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync
uv run ruff check .
