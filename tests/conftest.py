"""
Shared fixtures for the mqtt_llm.py test suite.

mqtt_llm.py is a script, not a package: nearly everything (argument
parsing, MQTT connect, the final shutdown_event.wait()) runs at import
time, in an unguarded module body. To unit-test the functions inside it
without hitting a real broker or blocking forever, `load_mqtt_llm()`
loads a fresh copy of the module per call with:

  - sys.argv patched to safe test arguments
  - mqtt.Client.connect / loop_start stubbed to no-ops (no real socket)
  - threading.Event.wait patched to return immediately (so the
    `shutdown_event.wait()` at the bottom of the script doesn't block
    the import forever, and the shutdown/cleanup code at the bottom
    runs once, synchronously, as part of loading)

Each call produces an independent module object (unique sys.modules
key, popped again immediately) so tests don't leak global state
(request_queue, active_responses, shutdown_event) into each other.

For the parts that are impractical or unsafe to mock convincingly
(paho's real network loop, a real MQTT broker, an HTTP server speaking
Ollama's streaming protocol), we run the real thing locally: a real
Mosquitto broker as a subprocess, and a small stdlib HTTP server that
mimics Ollama's /api/generate NDJSON streaming format.
"""
import contextlib
import importlib.util
import itertools
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import paho.mqtt.client as mqtt

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "mqtt_llm.py"

_module_counter = itertools.count()


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
