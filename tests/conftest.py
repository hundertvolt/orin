"""
Shared fixtures for the mqtt_llm.py and mqtt_telemetry.py test suites.

Both are scripts, not packages: nearly everything (argument parsing,
MQTT connect, the main loop) runs at import time, in an unguarded
module body. To unit-test the functions inside them without hitting a
real broker or blocking forever, `load_mqtt_llm()` / `load_mqtt_telemetry()`
load a fresh copy of the module per call with:

  - sys.argv patched to safe test arguments
  - mqtt.Client.connect / loop_start stubbed to no-ops (no real socket)
  - signal.signal() stubbed (only works on the main thread; exec_module()
    runs on a background thread here)
  - a module-specific trick to unblock whatever the script blocks on at
    the end of its main body (see each loader's docstring)

Each call produces an independent module object (unique sys.modules
key, popped again immediately) so tests don't leak global state into
each other.

For the parts that are impractical or unsafe to mock convincingly
(paho's real network loop, a real MQTT broker, an HTTP server speaking
Ollama's streaming protocol, the jetson-stats service), we run the real
thing locally where practical: a real Mosquitto broker as a subprocess,
a small stdlib HTTP server that mimics Ollama's /api/generate NDJSON
streaming format, and a minimal fake `jtop` package (real jetson-stats
needs actual Jetson hardware and isn't installable here).
"""
import contextlib
import datetime
import importlib.util
import itertools
import json
import socket
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "mqtt_llm.py"
TELEMETRY_MODULE_PATH = REPO_ROOT / "mqtt_telemetry.py"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_module_counter = itertools.count()
_telemetry_module_counter = itertools.count()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _patched_client_methods(method_mocks):
    with contextlib.ExitStack() as stack:
        for name, mock in method_mocks.items():
            stack.enter_context(patch.object(mqtt.Client, name, mock))
        yield


def load_mqtt_llm(argv, client_method_mocks=None):
    """Load a fresh, isolated instance of mqtt_llm.py for testing.

    argv: list of CLI args, e.g. ["--broker", "127.0.0.1"].
    client_method_mocks: dict of {method_name: MagicMock()} applied to
        mqtt.Client at the class level for the duration of the load.
        'connect' and 'loop_start' default to no-op mocks unless
        overridden, so import never touches a real socket.

    mqtt_llm.py blocks on shutdown_event.wait() as the last thing it
    does at import time. Rather than monkeypatching threading.Event
    (which also breaks Thread's own start()/join() synchronization,
    since Thread uses Event internally), we run exec_module() in a
    background thread, poll for the real shutdown_event to come into
    existence, and set() it - the same real signal handle_exit() would
    set. That lets the module's own shutdown/cleanup code run for
    real, synchronously, with real threading semantics intact.

    mqtt_llm.py also now blocks earlier, in wait_for_mqtt_connection(),
    until is_connected() is True or shutdown_event is set. Since
    connect()/loop_start() are stubbed, is_connected() never turns True
    on its own here, so shutdown_event.set() alone would only be
    noticed once connected_event.wait()'s own timeout elapses (5s) -
    handle_exit() sets both events together for exactly this reason, so
    we do too, to unblock that wait promptly instead of eating a real
    5-second delay on every load.
    """
    mocks = dict(client_method_mocks or {})
    mocks.setdefault("connect", MagicMock(return_value=None))
    mocks.setdefault("loop_start", MagicMock(return_value=None))

    mod_name = f"mqtt_llm_under_test_{next(_module_counter)}"
    spec = importlib.util.spec_from_file_location(mod_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)

    exec_errors = []

    def run():
        # signal.signal() only works on the main thread; exec_module() runs on
        # a background thread here, and real signal delivery is covered
        # separately by the subprocess-based integration tests.
        with patch.object(sys, "argv", ["mqtt_llm.py", *argv]), \
             patch("signal.signal"), \
             _patched_client_methods(mocks):
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                exec_errors.append(e)

    loader_thread = threading.Thread(target=run, name="mqtt-llm-test-loader", daemon=True)
    loader_thread.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not hasattr(module, "shutdown_event") and not exec_errors:
        time.sleep(0.005)

    if exec_errors:
        loader_thread.join(timeout=5)
        sys.modules.pop(mod_name, None)
        raise exec_errors[0]
    if not hasattr(module, "shutdown_event"):
        raise RuntimeError("mqtt_llm.py did not reach shutdown_event definition in time")

    module.shutdown_event.set()  # same trigger handle_exit() uses on SIGTERM/SIGINT
    module.connected_event.set()  # handle_exit() sets this too, to unblock wait_for_mqtt_connection()
    loader_thread.join(timeout=15)
    sys.modules.pop(mod_name, None)

    if loader_thread.is_alive():
        raise RuntimeError("mqtt_llm.py did not finish its shutdown sequence within the timeout")
    if exec_errors:
        raise exec_errors[0]

    # The module's own shutdown sequence has already run once (that's how we
    # unblocked the import); reset the flag so callers get a module that
    # looks like a freshly-started, not-yet-shutting-down service.
    module.shutdown_event.clear()

    return module


@pytest.fixture
def loaded_module():
    """A safely-loaded mqtt_llm module instance, default args, no real I/O."""
    return load_mqtt_llm(["--broker", "127.0.0.1", "--loglevel", "DEBUG"])


# ------------------------------
# Real local Mosquitto broker
# ------------------------------

@pytest.fixture
def mosquitto_broker(tmp_path):
    port = free_port()
    conf = tmp_path / "mosquitto.conf"
    conf.write_text(
        f"listener {port} 127.0.0.1\n"
        "allow_anonymous true\n"
        "persistence false\n"
        "log_type error\n"
    )
    proc = subprocess.Popen(
        ["mosquitto", "-c", str(conf)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        proc.wait()
        raise RuntimeError("mosquitto did not start listening in time")

    yield port

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def restart_mosquitto(proc_holder, tmp_path, port):
    """Kill and respawn mosquitto on the same port, simulating a broker restart."""
    conf = tmp_path / "mosquitto.conf"
    proc_holder["proc"].kill()
    proc_holder["proc"].wait()
    proc = subprocess.Popen(
        ["mosquitto", "-c", str(conf)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc_holder["proc"] = proc
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("mosquitto did not come back up in time")


# ------------------------------
# Fake local Ollama HTTP server
# ------------------------------

class _ScenarioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _OllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            self.last_request = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.last_request = None
        self.server.received_requests.append(self.last_request)
        self.server.last_connection = self.connection  # lets tests force-close a hung stream

        scenario = self.server.scenario

        if scenario == "http_error":
            error_body = b"model not found"
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self.wfile.write(error_body)
            return

        if scenario == "hang":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write((json.dumps({"response": "partial-"}) + "\n").encode())
            self.wfile.flush()
            time.sleep(self.server.hang_seconds)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for chunk in self.server.chunks:
            self.wfile.write((json.dumps(chunk) + "\n").encode())
            self.wfile.flush()
            if self.server.chunk_delay:
                time.sleep(self.server.chunk_delay)


@pytest.fixture
def fake_ollama():
    server = _ScenarioHTTPServer(("127.0.0.1", 0), _OllamaHandler)
    server.scenario = "success"
    server.chunks = [
        {"response": "Hello, "},
        {"response": "world!", "thinking": "greeting"},
        {"response": "", "done": True, "done_reason": "stop", "total_duration": 123},
    ]
    server.chunk_delay = 0
    server.hang_seconds = 0
    server.received_requests = []
    server.last_connection = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


# ------------------------------
# mqtt_telemetry.py loader + fake jtop
# ------------------------------

BASELINE_JTOP_STATS = {
    "CPU1": 5.0, "CPU2": 8.0, "CPU3": 3.0, "CPU4": 6.0,
    "Temp cpu": 45.0, "Temp gpu": 44.0, "Temp tj": -256,  # -256 = sensor absent
    "RAM": 0.35, "SWAP": 0.0, "GPU": 12.5,
    "uptime": datetime.timedelta(seconds=3725),
    "Power TOT": 4200,
}
BASELINE_JTOP_FAN = {"pwmfan": {"speed": [60], "rpm": [1500]}}


class FakeJetson:
    """Stand-in for a jtop instance, for calling publish_telemetry() directly.

    ok()/close() are no-ops here, matching the real jtop client's shape:
    ok() is what actually surfaces a lost connection (see
    RaisingFakeJetson), not .stats/.fan themselves.
    """

    def __init__(self, stats=None, fan=None):
        self._stats = stats if stats is not None else {}
        self._fan = fan if fan is not None else {}

    def ok(self, spin=False):
        return True

    def close(self):
        pass

    @property
    def stats(self):
        return self._stats

    @property
    def fan(self):
        return self._fan


class RaisingFakeJetson:
    """A jtop stand-in whose .ok() fails, for exception-path tests.

    This mirrors the real jtop client: a lost connection is detected by
    its background reader thread, stored, and only surfaced when the
    caller calls ok() - .stats/.fan themselves are plain in-memory reads
    that never raise for a connectivity reason (see jtop.py upstream,
    jtop.ok()/jtop._get_data()).
    """

    def __init__(self, error):
        self._error = error

    def ok(self, spin=False):
        raise self._error

    def close(self):
        pass

    @property
    def stats(self):
        return {}

    @property
    def fan(self):
        return {}


def _install_fake_jtop(stats=None, fan=None, enter_error=None, enter_error_times=None, ok_error=None, ok_error_after=0):
    """Install a minimal fake `jtop` module into sys.modules.

    mqtt_telemetry.py does `from jtop import jtop`; the real jetson-stats
    package needs actual Jetson hardware and the jetson_stats service, so
    it isn't usable here. This fakes just the shape the script relies on:
    a context manager exposing .ok()/.stats/.fan/.close(). Returns a dict
    tracking how many times __enter__ was called, so tests can confirm
    jtop() was actually invoked (and how many times, across retries).

    enter_error, given alone, raises on every __enter__() call - an
    outage that never clears, for testing that the caller keeps retrying
    instead of crashing. Pass enter_error_times as well to only raise for
    that many attempts and then succeed normally, for testing that a
    later retry actually recovers.

    ok_error, given, makes ok(spin=True) raise it once it's been called
    more than ok_error_after times *on a given fake jtop instance* - a
    mid-run failure after some number of successful publishes, matching
    the real client's failure-detection path (see RaisingFakeJetson). The
    call counter is per-instance, not global, since a fresh jtop()
    instance after a retry has its own, initially-healthy state in the
    real client too.
    """
    enter_count = {"n": 0}

    class FakeJtopContextManager:
        def __init__(self):
            self._ok_calls = 0

        def __enter__(self):
            enter_count["n"] += 1
            if enter_error is not None and (enter_error_times is None or enter_count["n"] <= enter_error_times):
                raise enter_error
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def ok(self, spin=False):
            self._ok_calls += 1
            if ok_error is not None and self._ok_calls > ok_error_after:
                raise ok_error
            return True

        def close(self):
            pass

        @property
        def stats(self):
            return stats if stats is not None else {}

        @property
        def fan(self):
            return fan if fan is not None else {}

    fake_module = types.ModuleType("jtop")
    fake_module.jtop = FakeJtopContextManager
    sys.modules["jtop"] = fake_module
    return enter_count


def load_mqtt_telemetry(
    argv,
    client_method_mocks=None,
    jtop_stats=None,
    jtop_fan=None,
    jtop_enter_error=None,
    jtop_enter_error_times=None,
    jtop_ok_error=None,
    jtop_ok_error_after=0,
    sleep_calls_before_exit=1,
):
    """Load a fresh, isolated instance of mqtt_telemetry.py for testing.

    Mirrors load_mqtt_llm(): connect()/loop_start() are stubbed,
    signal.signal() is stubbed, and exec_module() runs on a background
    thread. mqtt_telemetry.py has no shutdown_event to trigger
    externally - instead its main loop blocks in time.sleep() (either
    between publishes, or between jtop retry attempts), so time.sleep is
    patched to raise SystemExit once it's been called
    sleep_calls_before_exit times: the same effect handle_exit() has in
    production when a signal arrives during one of those sleeps. With
    the default of 1, that's the very first sleep - letting the module
    publish once for real (against the fake jtop given here) and then
    run its own shutdown/cleanup code, exactly like a real SIGTERM
    would. Pass a higher value to let the module run through more
    sleeps for real first (e.g. surviving one jtop retry delay before
    the module is asked to shut down), for tests that need to observe
    what happens after a retry, not just before it.

    mqtt_telemetry.py now also refuses to call jtop() at all until
    wait_for_mqtt_connection() sees is_connected() return True. Since
    connect()/loop_start() are stubbed, on_connect() never fires
    naturally and is_connected() would default to the real (never
    truly connected) paho state - which would make that wait loop
    forever for every test, not just the ones specifically interested
    in it. So is_connected defaults to True here (connect()/loop_start()
    already default to "succeeded" no-ops; this keeps that same
    "happy path by default" story), which resolves the wait on its very
    first check without needing threading.Event.wait at all. Pass a
    stateful is_connected mock (e.g. true-then-false, for "was connected,
    then dropped by shutdown time") to override this for a specific test.

    threading.Event.wait is still patched to return immediately, as a
    safety net for any test that does exercise the wait loop with a
    genuinely-not-yet-connected is_connected mock (no other code here
    starts real threads during this call, so this doesn't hit the
    Thread-internals hazard that ruled out the same trick for
    shutdown_event in load_mqtt_llm()). This has no effect on what
    is_connected() itself reports.

    If jtop_enter_error is given, jtop().__enter__() raises it instead
    (see _install_fake_jtop for enter_error_times). If jtop_ok_error is
    given, ok() raises it instead once called more than
    jtop_ok_error_after times on a given jtop instance, simulating a
    mid-run failure after a successful open (see _install_fake_jtop).
    mqtt_telemetry.py's main loop now retries a jtop() failure - whether
    at open or mid-run - rather than propagating it, so either of these
    by itself no longer crashes the module; it just makes the module log
    and retry, same as any other jtop outage.
    """
    mocks = dict(client_method_mocks or {})
    mocks.setdefault("connect", MagicMock(return_value=None))
    mocks.setdefault("loop_start", MagicMock(return_value=None))
    mocks.setdefault("is_connected", MagicMock(return_value=True))

    enter_count = _install_fake_jtop(
        stats=jtop_stats, fan=jtop_fan,
        enter_error=jtop_enter_error, enter_error_times=jtop_enter_error_times,
        ok_error=jtop_ok_error, ok_error_after=jtop_ok_error_after,
    )

    mod_name = f"mqtt_telemetry_under_test_{next(_telemetry_module_counter)}"
    spec = importlib.util.spec_from_file_location(mod_name, TELEMETRY_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)

    exec_errors = []

    sleep_calls = {"n": 0}

    def fake_sleep(seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= sleep_calls_before_exit:
            raise SystemExit

    def run():
        with patch.object(sys, "argv", ["mqtt_telemetry.py", *argv]), \
             patch("signal.signal"), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch("threading.Event.wait", return_value=True), \
             _patched_client_methods(mocks):
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                exec_errors.append(e)

    loader_thread = threading.Thread(target=run, name="mqtt-telemetry-test-loader", daemon=True)
    loader_thread.start()
    loader_thread.join(timeout=15)

    sys.modules.pop(mod_name, None)
    sys.modules.pop("jtop", None)

    if loader_thread.is_alive():
        raise RuntimeError("mqtt_telemetry.py did not finish within the timeout")
    if exec_errors:
        raise exec_errors[0]

    module.jtop_enter_count = enter_count["n"]
    return module


@pytest.fixture
def loaded_telemetry_module():
    """A safely-loaded mqtt_telemetry module instance: connect()/loop_start()
    stubbed, one publish cycle executed against a fake jtop with baseline
    stats, then shut down via a simulated signal - same as a real SIGTERM
    arriving during the interval sleep."""
    return load_mqtt_telemetry(
        ["--broker", "127.0.0.1", "--loglevel", "DEBUG"],
        jtop_stats=BASELINE_JTOP_STATS,
        jtop_fan=BASELINE_JTOP_FAN,
    )
