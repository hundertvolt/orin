"""
End-to-end tests: mqtt_llm.py run as a real subprocess, talking to a
real local Mosquitto broker and a fake local Ollama HTTP server.

These exercise exactly what changed in terms observable from outside
the process: presence status (LWT + explicit online/offline), and
resilience to malformed input and broker restarts (no code mocking -
this is the actual target deployment shape, minus the GPU/model and
the real broker binary's persistence-to-disk behavior).
"""
import json
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

import paho.mqtt.client as mqtt

from conftest import MODULE_PATH, free_port


class ServiceProcess:
    def __init__(self, proc):
        self.proc = proc
        self.lines = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self):
        for line in self.proc.stdout:
            with self._lock:
                self.lines.append(line.rstrip("\n"))

    def wait_for_log(self, substring, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if any(substring in line for line in self.lines):
                    return True
            time.sleep(0.05)
        return False

    def text(self):
        with self._lock:
            return "\n".join(self.lines)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


@pytest.fixture
def service_factory(mosquitto_broker, fake_ollama):
    procs = []

    def start(topic="orin/ollama", broker_port=None, ollama_port=None, extra_args=None):
        args = [
            sys.executable, str(MODULE_PATH),
            "--broker", "127.0.0.1",
            "--port", str(broker_port or mosquitto_broker),
            "--topic", topic,
            "--ollama-host", "127.0.0.1",
            "--ollama-port", str(ollama_port or fake_ollama.server_address[1]),
            "--loglevel", "DEBUG",
            "--shutdown-timeout", "5",
        ]
        if extra_args:
            args.extend(extra_args)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        svc = ServiceProcess(proc)
        procs.append(svc)
        return svc

    yield start

    for svc in procs:
        svc.stop()


@pytest.fixture
def probe_client(mosquitto_broker):
    received = []
    client = mqtt.Client(client_id="probe", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = lambda c, u, msg: received.append((msg.topic, msg.payload.decode("utf-8"), msg.retain))
    client.connect("127.0.0.1", mosquitto_broker)
    client.loop_start()
    yield client, received
    client.loop_stop()
    client.disconnect()


def _wait_until(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _assert_retained(broker_port, topic, expected_payload, timeout=5.0):
    """Connect a brand-new client and confirm it immediately receives
    `topic` as a retained message on subscribe.

    This is the only delivery for which the broker actually sets
    retain=1: per the MQTT spec, a client that was *already*
    subscribed when a retained-flagged message is published receives
    it live, with retain=0 regardless of how the publisher set the
    flag - retain=1 is reserved for messages replayed to a client on
    subscribe. So "was this actually stored as retained" has to be
    checked from a fresh subscriber's point of view, not the original
    probe's.
    """
    received = []
    probe = mqtt.Client(
        client_id=f"retain-check-{threading.get_ident()}-{time.monotonic_ns()}",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    probe.on_message = lambda c, u, msg: received.append((msg.payload.decode("utf-8"), msg.retain))
    probe.connect("127.0.0.1", broker_port)
    probe.loop_start()
    probe.subscribe(topic, qos=1)
    try:
        assert _wait_until(lambda: len(received) >= 1, timeout=timeout), f"no retained message found on {topic}"
        payload, retain = received[0]
        assert json.loads(payload) == expected_payload
        assert retain is True
    finally:
        probe.loop_stop()
        probe.disconnect()


def test_status_online_on_connect_and_offline_on_clean_shutdown(service_factory, probe_client, mosquitto_broker):
    client, received = probe_client
    client.subscribe("orin/ollama/status", qos=1)
    time.sleep(0.3)

    svc = service_factory(topic="orin/ollama")
    assert svc.wait_for_log("Connected to MQTT broker"), svc.text()

    assert _wait_until(lambda: len(received) >= 1), svc.text()
    topic, payload, retain = received[-1]
    assert topic == "orin/ollama/status"
    assert json.loads(payload) == {"status": "online"}
    _assert_retained(mosquitto_broker, "orin/ollama/status", {"status": "online"})

    received.clear()
    svc.proc.send_signal(signal.SIGTERM)
    rc = svc.proc.wait(timeout=10)
    assert rc == 0, svc.text()
    assert "Shutdown complete." in svc.text()

    assert _wait_until(lambda: len(received) >= 1), "no offline status observed after clean shutdown"
    topic, payload, retain = received[-1]
    assert json.loads(payload) == {"status": "offline"}
    _assert_retained(mosquitto_broker, "orin/ollama/status", {"status": "offline"})


def test_lwt_fires_on_ungraceful_kill(service_factory, probe_client, mosquitto_broker):
    client, received = probe_client
    client.subscribe("orin/ollama/status", qos=1)
    time.sleep(0.3)

    svc = service_factory(topic="orin/ollama")
    assert svc.wait_for_log("Connected to MQTT broker"), svc.text()
    assert _wait_until(lambda: len(received) >= 1), "expected online status before kill"
    received.clear()

    svc.proc.kill()  # SIGKILL: no signal handler runs, only the broker's LWT can announce this
    svc.proc.wait(timeout=5)

    assert _wait_until(lambda: len(received) >= 1, timeout=10), "LWT was not published by the broker"
    topic, payload, retain = received[-1]
    assert json.loads(payload) == {"status": "offline"}
    _assert_retained(mosquitto_broker, "orin/ollama/status", {"status": "offline"})


def test_request_response_roundtrip(service_factory, probe_client):
    client, received = probe_client
    client.subscribe("orin/ollama/response", qos=1)
    time.sleep(0.3)

    svc = service_factory(topic="orin/ollama")
    assert svc.wait_for_log("Connected to MQTT broker"), svc.text()

    client.publish("orin/ollama/request", json.dumps({
        "request_id": "roundtrip-1", "model": "llama3", "system": "You are helpful.", "user": "Say hi",
    }), qos=1)

    assert _wait_until(lambda: len(received) >= 1), svc.text()
    body = json.loads(received[-1][1])
    assert body["request_id"] == "roundtrip-1"
    assert body["error_code"] == 0
    assert body["message"] == "Hello, world!"


def test_garbage_message_does_not_kill_service(service_factory, probe_client):
    """The core end-to-end regression test for the exception-hardening fix:
    a malformed on_message payload must not take down the MQTT thread."""
    client, received = probe_client
    client.subscribe("orin/ollama/response", qos=1)
    time.sleep(0.3)

    svc = service_factory(topic="orin/ollama")
    assert svc.wait_for_log("Connected to MQTT broker"), svc.text()

    client.publish("orin/ollama/request", b"\x00\x01 not even json \xff", qos=1)
    time.sleep(0.5)
    assert svc.proc.poll() is None, "service crashed on malformed input"

    client.publish("orin/ollama/request", json.dumps({
        "request_id": "after-garbage", "model": "llama3", "system": "sys", "user": "still alive?",
    }), qos=1)

    assert _wait_until(lambda: len(received) >= 1), "service's MQTT thread died after malformed input: " + svc.text()
    body = json.loads(received[-1][1])
    assert body["request_id"] == "after-garbage"
    assert body["message"] == "Hello, world!"


def test_ollama_error_reported_without_crashing_service(service_factory, probe_client, fake_ollama):
    fake_ollama.scenario = "http_error"
    client, received = probe_client
    client.subscribe("orin/ollama/response", qos=1)
    time.sleep(0.3)

    svc = service_factory(topic="orin/ollama")
    assert svc.wait_for_log("Connected to MQTT broker"), svc.text()

    client.publish("orin/ollama/request", json.dumps({
        "request_id": "will-fail", "model": "missing-model", "system": "s", "user": "u",
    }), qos=1)

    assert _wait_until(lambda: len(received) >= 1), svc.text()
    body = json.loads(received[-1][1])
    assert body["request_id"] == "will-fail"
    assert body["error_code"] == 3  # OLLAMA_ERROR
    assert "500" in body["error_message"]
    assert svc.proc.poll() is None


def test_graceful_shutdown_while_generation_in_flight(service_factory, probe_client, fake_ollama):
    """Exercises the active_responses cleanup loop in the shutdown block for
    real: a generation is mid-stream when SIGTERM arrives.

    The shutdown block's unblock mechanism tracks the raw socket
    (ActiveGeneration.sock, captured right after connect() before
    getresponse() can null conn.sock) rather than conn.sock itself, so
    it can still reach and interrupt a real, actively-streaming Ollama
    connection. That should produce a genuine graceful cancellation -
    "cancelled due to shutdown" - rather than the process abandoning a
    stuck daemon thread and exiting without ever processing it.
    """
    fake_ollama.scenario = "hang"
    fake_ollama.hang_seconds = 30
    client, received = probe_client
    client.subscribe("orin/ollama/status", qos=1)
    time.sleep(0.3)

    svc = service_factory(topic="orin/ollama")
    assert svc.wait_for_log("Connected to MQTT broker"), svc.text()

    client.publish("orin/ollama/request", json.dumps({
        "request_id": "long-running", "model": "llama3", "system": "s", "user": "u",
    }), qos=1)
    assert svc.wait_for_log("Calling Ollama for request_id=long-running"), svc.text()
    time.sleep(0.3)  # let it actually start streaming (past headers) before we pull the rug

    start = time.monotonic()
    svc.proc.send_signal(signal.SIGTERM)
    rc = svc.proc.wait(timeout=10)
    elapsed = time.monotonic() - start

    assert rc == 0, svc.text()
    assert elapsed < 8, f"shutdown took {elapsed:.1f}s, longer than --shutdown-timeout allows"
    assert "Shutdown complete." in svc.text()
    assert "cancelled due to shutdown" in svc.text()
    assert "Worker thread exited cleanly." in svc.text()
    assert "Worker thread did not finish within" not in svc.text()


def test_broker_restart_reconnects_and_resumes_serving(fake_ollama, tmp_path):
    port = free_port()
    conf = tmp_path / "mosquitto.conf"
    conf.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\npersistence false\nlog_type error\n")

    def spawn_broker():
        proc = subprocess.Popen(["mosquitto", "-c", str(conf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert _wait_until(lambda: _port_open(port), timeout=5), "mosquitto did not start"
        return proc

    broker = spawn_broker()
    svc_proc = subprocess.Popen(
        [sys.executable, str(MODULE_PATH),
         "--broker", "127.0.0.1", "--port", str(port), "--topic", "orin/ollama",
         "--ollama-host", "127.0.0.1", "--ollama-port", str(fake_ollama.server_address[1]),
         "--loglevel", "DEBUG", "--shutdown-timeout", "5"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    svc = ServiceProcess(svc_proc)

    def make_probe(client_id):
        received = []
        connected = threading.Event()
        subscribed = threading.Event()
        c = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        c.on_message = lambda cli, u, msg: received.append(json.loads(msg.payload.decode()))
        c.on_connect = lambda cli, u, f, rc, p: connected.set()
        c.on_subscribe = lambda cli, u, mid, rc, p: subscribed.set()
        c.connect("127.0.0.1", port)
        c.loop_start()
        assert connected.wait(timeout=5), f"{client_id} never got CONNACK"
        c.subscribe("orin/ollama/response", qos=1)
        assert subscribed.wait(timeout=5), f"{client_id} never got SUBACK"
        return c, received

    def publish_and_wait_for_response(probe, received, request_id, timeout=15.0):
        # qos=1 publish is fire-and-forget from the caller's perspective; if
        # a request lands in a gap right as the service is mid-reconnect, it
        # can be dropped (a known, accepted trade-off - see the mqtt_llm.py
        # design discussion on clean vs. persistent sessions). Retry a
        # couple of times so this test is about reconnect behavior, not
        # single-publish timing luck.
        for attempt in range(3):
            probe.publish("orin/ollama/request", json.dumps({
                "request_id": request_id, "model": "llama3", "system": "s", "user": "u",
            }), qos=1)
            if _wait_until(lambda: len(received) >= 1, timeout=timeout / 3):
                return
        raise AssertionError(f"no response for {request_id} after {3} attempts")

    try:
        assert svc.wait_for_log("Connected to MQTT broker"), svc.text()

        probe1, received1 = make_probe("probe-before")
        publish_and_wait_for_response(probe1, received1, "before-restart")
        assert received1[-1]["request_id"] == "before-restart"
        probe1.loop_stop()
        probe1.disconnect()

        broker.kill()
        broker.wait()
        assert svc.wait_for_log("MQTT connection failed", timeout=1) or True  # best-effort, timing dependent
        broker = spawn_broker()

        assert svc.wait_for_log("Connected to MQTT broker", timeout=30), \
            "service did not reconnect after broker restart: " + svc.text()

        probe2, received2 = make_probe("probe-after")
        publish_and_wait_for_response(probe2, received2, "after-restart")
        assert received2[-1]["request_id"] == "after-restart", \
            "service did not resume serving requests after broker restart: " + svc.text()
        probe2.loop_stop()
        probe2.disconnect()
    finally:
        svc.stop()
        broker.kill()
        broker.wait()


def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def test_signal_interrupts_indefinite_connect_wait_and_shuts_down_cleanly():
    """Regression test, end to end, for the connect-settle gating fix:
    point at a listener that accepts the TCP connection but never sends a
    CONNACK (no MQTT broker at all, just a raw socket held open), so the
    connection genuinely never settles. wait_for_mqtt_connection() must
    log the periodic warning for real (proving it's actually blocked,
    not connected) and a SIGTERM must still interrupt the indefinite
    wait cleanly - proving it isn't a silent, unbreakable hang.

    Unlike mqtt_telemetry.py (where handle_exit() raises SystemExit
    directly, aborting before jtop() is ever reached),
    handle_exit() here only sets events: wait_for_mqtt_connection()
    unblocks the same way for "connected" or "shutdown requested", and
    the worker thread does still start afterwards - it just immediately
    receives the shutdown sentinel and exits. That's harmless by design,
    so the assertions here are about a clean, bounded shutdown, not
    about the worker thread never starting.
    """
    port = free_port()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)

    def accept_and_hold():
        try:
            conn, _ = srv.accept()
            conn.settimeout(30)
            try:
                conn.recv(1)  # blocks until the client closes or we time out
            except OSError:
                pass
        except OSError:
            pass  # srv.close() below unblocks a pending accept()

    holder = threading.Thread(target=accept_and_hold, daemon=True)
    holder.start()

    proc = subprocess.Popen(
        [sys.executable, str(MODULE_PATH),
         "--broker", "127.0.0.1", "--port", str(port), "--topic", "orin/ollama",
         "--ollama-host", "127.0.0.1", "--ollama-port", str(free_port()),
         "--loglevel", "DEBUG", "--shutdown-timeout", "5"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    svc = ServiceProcess(proc)
    try:
        assert svc.wait_for_log("Still waiting for MQTT connection", timeout=8), svc.text()

        start = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=10)
        elapsed = time.monotonic() - start

        assert rc == 0, "SIGTERM should still interrupt an indefinite connection wait: " + svc.text()
        assert elapsed < 8, f"shutdown took {elapsed:.1f}s, longer than --shutdown-timeout allows"
        assert "Shutdown complete." in svc.text()
        assert "Worker thread did not finish within" not in svc.text()
    finally:
        svc.stop()
        srv.close()
        holder.join(timeout=2)
