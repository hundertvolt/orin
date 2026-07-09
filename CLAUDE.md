# CLAUDE.md

Guidance for Claude Code when working in this repo. See `README.md` first for
what the project does and how to run it; this file is about how to work on it.

## What this is

Two independent, standalone MQTT services for an NVIDIA Jetson Orin Nano —
`mqtt_llm.py` (MQTT↔Ollama bridge) and `mqtt_telemetry.py` (jetson-stats →
MQTT publisher). Not a package, not a library: each file is meant to be run
directly with `python3 <script>.py`, deployed as its own systemd unit. Keep
them that way — do not merge them, split them into modules, or turn this into
an installable package unless explicitly asked.

Target runtime is JetPack 6.x / Python 3.10 on real Jetson hardware. `jtop`
(jetson-stats) only installs there, so `mqtt_telemetry.py` cannot actually run
off-device — but the whole toolchain (lint, type check, tests) works on any
Linux machine via `uv sync`, because the test suite fakes out the Jetson-only
bits (see `tests/conftest.py`, `tests/fixtures/`).

## Design invariants (read before touching either script)

Both scripts were hardened through a dedicated production-readiness pass.
The patterns below are load-bearing — if a change seems to require breaking
one of them, stop and reconsider rather than routing around it:

- **Unhandled exceptions must not kill the process.** Every paho-mqtt
  callback (`on_connect`, `on_disconnect`, `on_message`) wraps its body in
  `try/except Exception` and logs instead of raising — paho re-raises
  exceptions that escape a callback, which kills its network loop thread
  for good.
- **Presence/LWT semantics**: both scripts `will_set()` an "offline" payload
  as the MQTT Last Will (covers crashes, `kill -9`, power loss), *and*
  explicitly publish "offline" during a clean shutdown — because a graceful
  `disconnect()` cancels LWT delivery, so a clean exit needs its own
  explicit offline publish to still announce anything.
- **Connection-race avoidance**: `client.connect()` only sends the CONNECT
  packet; the CONNACK that flips `is_connected()` to `True` arrives
  asynchronously on the network loop thread. Both scripts block on
  `wait_for_mqtt_connection()` (an `Event`-based wait, not polling) before
  doing anything that assumes a live connection — see the docstrings on
  that function in each file for the exact race it closes.
- **Uncaught-exceptions-should-crash, on purpose, at the top level.** The
  outermost `try/except` blocks around startup (`client.connect`,
  `loop_start`, worker thread start) deliberately `raise` after logging, so
  systemd restarts the service. Don't swallow these.
- **Graceful shutdown**: `SIGTERM`/`SIGINT` handlers set an `Event` rather
  than acting directly; the main thread parks on that event and then runs a
  single, deterministic shutdown sequence (drain/cancel in-flight work,
  publish offline, `loop_stop()`, `disconnect()`). In `mqtt_llm.py`, in-flight
  Ollama HTTP requests are interrupted by directly `shutdown(SOCK_RDWR)`-ing
  the captured raw socket (see `ActiveGeneration` docstring for why
  `HTTPConnection.sock` can't be relied on once streaming starts), then
  `worker_thread.join(timeout=SHUTDOWN_TIMEOUT)` with a daemon-thread
  fallback so shutdown can't hang forever.
- **Locking**: `active_responses_lock` in `mqtt_llm.py` protects
  `active_responses` end-to-end — registering a call and checking
  `shutdown_event` happen under the *same* lock acquisition specifically to
  close a register-vs-shutdown race (see the comment at the top of
  `stream_ollama_generate`). Preserve that pairing if you touch this path.
- **CLI validation happens at argparse time** (`_port_type`,
  `_positive_int`/`_positive_float`), not deep in the code — bad input
  should fail immediately and clearly at startup, not surface later as an
  unrelated exception (e.g. `time.sleep()` raising on a negative interval).
- **A dependency outage retries in-process; it does not crash the service.**
  Both scripts treat "Ollama/jtop is temporarily unavailable" as an expected,
  self-clearing condition, not a reason to let systemd restart the whole
  process — a full restart is the last-resort fallback for genuinely
  unexpected bugs, not the intended recovery path for a dependency bouncing
  during an update. In `mqtt_llm.py` this falls out naturally: a fresh
  `http.client.HTTPConnection` is created per request, so every Ollama
  failure (refused, reset, timeout, bad data) just fails that one request
  and the worker loop moves on. In `mqtt_telemetry.py`, `jtop()` failing to
  open *and* jtop breaking mid-run are handled identically — discard
  whatever jtop state exists and retry a fresh `jtop()` instance after
  `JTOP_RETRY_DELAY`, rather than distinguishing "never connected" from
  "connection broke". **Read `jtop.py` upstream (jetson-stats on PyPI, pure
  Python, `pip download jetson-stats` works off-device) before changing this
  path** — it's not what it looks like: `jetson.stats`/`.fan` are plain
  in-memory reads with zero I/O (they cannot hang), but they also never
  raise for a lost connection — jtop's background reader thread catches
  that internally and only surfaces it if the caller calls `jetson.ok()`.
  `publish_telemetry()` calls `jetson.ok(spin=True)` specifically for this;
  removing it silently brings back "stale data forever, no error, no retry"
  for any mid-run outage. Also note `jtop.__exit__` does **not** stop the
  background thread or close the connection — the retry loop calls
  `jetsonTop.close()` itself in a `finally`, or every retry attempt leaks
  one. A jtop/Ollama outage is also reported on the telemetry/status
  payload itself (`status: "offline"`, real non-zero `heartbeat`, sensor
  fields `null` for jtop) so subscribers see the degradation live, distinct
  from the `heartbeat: 0` LWT/shutdown message which means the process
  itself is gone.

If you extend either script, keep new code inside these same invariants
(caught-and-logged callback errors, explicit offline publish on clean exit,
event-based waits instead of polling, fail-fast CLI validation, retry
dependency outages in-process rather than crashing).

## Code style

- No line-length limit is enforced (`E501` is off, `line-length = 320` in
  `pyproject.toml`). Line breaks in this codebase are chosen by hand for
  readability, not by a formatter. **Never run `ruff format`** and don't
  introduce automated line-wrapping — this was a deliberate, explicit
  decision (see the comment above `[tool.ruff]` in `pyproject.toml`). If a
  tool or auto-fix wants to reflow lines, drop it or scope it down rather
  than accepting the reflow.
- Comments explain *why*, not *what* — matching the docstrings/comments
  already in both scripts (e.g. the `ActiveGeneration` and
  `wait_for_mqtt_connection` docstrings). Don't add comments that restate
  the code.
- Modern typing: `X | None` not `Optional[X]`, `dict[...]`/`list[...]` not
  `Dict[...]`/`List[...]` (enforced by Ruff's `UP` rules).
- paho-mqtt callbacks must use paho's real typed signatures
  (`mqtt.ConnectFlags`, `mqtt.DisconnectFlags`, `mqtt.ReasonCode`,
  `mqtt.Properties | None`) — not generic `Dict`/`int`/`Any`. paho ships
  inline types (`py.typed`), so this is checked for real, not just stubbed.

## Checks: config and how to run them

All tool config lives in the single `pyproject.toml` — no `mypy.ini`,
`pytest.ini`, or `requirements*.txt` files; keep it that way, don't reopen
that split. Dependencies are managed with `uv` (`[project.dependencies]` +
`[dependency-groups].dev`; `uv.lock` is committed).

```bash
./scripts/ci.sh          # everything CI runs, in order: lint, type check, tests
./scripts/lint.sh        # ruff check (lint only, see style note above)
./scripts/typecheck.sh   # mypy
./scripts/test.sh        # pytest — extra args pass through, e.g. -k foo -v
```

Each script starts with `uv sync`, so a venv is provisioned automatically —
no manual setup beyond having `uv` installed. `.python-version` pins Python
3.10 to match the JetPack 6.x target. Integration tests spawn a real local
Mosquitto broker as a subprocess, so `mosquitto` must be on `PATH`.
`.github/workflows/ci.yml` runs the exact same three scripts via
`astral-sh/setup-uv`, so local and CI behavior can't drift apart — if a check
needs to change, change the script (or `pyproject.toml`), not the workflow
file, and vice versa.

**mypy trap already fixed once, don't reintroduce it**: `scripts/typecheck.sh`
does `unset MYPYPATH` before `uv run mypy`. A global `MYPYPATH` set for an
unrelated project (e.g. pointing at MicroPython stub typings, which define
cut-down `socket`/`time` modules) is honored *additively* by mypy regardless
of which project config is discovered, and can shadow real stdlib typeshed
names — producing confusing false positives like `"socket" has no attribute
"shutdown"` on perfectly ordinary code. If mypy ever reports a stdlib type as
missing an obviously-real attribute, suspect environment leakage before
suspecting the code.

mypy config (`[tool.mypy]` in `pyproject.toml`) is "stricter than default,
short of `--strict`": untyped/incomplete defs are disallowed and Optional/Any
handling is checked, but noisier strict-mode-only checks are left off as
low-value for two standalone scripts. `jtop.*` is the only
`ignore_missing_imports` override (it's Jetson-only and can't be installed or
stubbed off-device); paho-mqtt and pydantic are fully typed already.

## Tests

`tests/` covers both scripts: callback unit tests, shutdown-race tests, and
integration tests against a real local Mosquitto broker plus a fake Ollama
HTTP server / faked `jtop` module (`tests/conftest.py`, `tests/fixtures/`).
When changing either script's behavior, extend the corresponding test file
rather than adding a new one, unless the change is a genuinely new concern —
follow the existing `test_<script>_<concern>.py` naming.

## Workflow notes

- This repo develops on `claude/mqtt-llm-review-8vgag9` against `main`; PRs
  get merged from there and the branch gets restarted from `main` for the
  next round of work (per the standing branch-restart protocol — not
  specific to this codebase, just how this repo's sessions have been run).
- Rely on GitHub webhook activity for PR feedback, not polling or scheduled
  check-ins — that was an explicit standing preference from earlier work on
  this repo.
- Before considering any change to either script done, run `./scripts/ci.sh`
  locally and let it pass (lint + type check + full test suite) — this has
  been the bar for every change so far.
