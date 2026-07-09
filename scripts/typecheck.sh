#!/usr/bin/env bash
# Runs the same type check as the CI "mypy" job.
#
# MYPYPATH is unset so a stray global value (e.g. pointing at a stub
# directory for an unrelated project) can't shadow stdlib modules like
# socket/time/threading with an incomplete set of names - which shows up
# as mypy reporting well-known stdlib members as missing.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset MYPYPATH

uv sync
uv run mypy
