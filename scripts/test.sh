#!/usr/bin/env bash
# Runs the same test suite as the CI "pytest" job. Works on any Linux
# machine (see [dependency-groups] in pyproject.toml) - mqtt_telemetry.py
# and mqtt_llm.py themselves only run natively on the Orin target, but the
# tests fake out the Jetson-only bits. Extra args are passed through to
# pytest, e.g. `scripts/test.sh -k stream_ollama -v`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v mosquitto >/dev/null 2>&1; then
    echo "warning: 'mosquitto' binary not found on PATH - the integration" >&2
    echo "         tests in tests/test_integration.py and" >&2
    echo "         tests/test_telemetry_integration.py will fail to start a" >&2
    echo "         broker. Install it first, e.g. 'apt install mosquitto'." >&2
fi

uv sync
uv run pytest "$@"
