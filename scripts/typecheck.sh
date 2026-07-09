#!/usr/bin/env bash
# Runs the same type check as the CI "mypy" job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync
uv run mypy
