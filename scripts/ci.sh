#!/usr/bin/env bash
# Runs the entire CI pipeline locally: lint, type check, then the test
# suite - same checks, same order, as .github/workflows/ci.yml.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

scripts/lint.sh
scripts/typecheck.sh
scripts/test.sh
