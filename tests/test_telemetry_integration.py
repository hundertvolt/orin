"""
End-to-end tests: mqtt_telemetry.py run as a real subprocess against a
real local Mosquitto broker, using an on-disk fake `jtop` package (real
jetson-stats needs actual Jetson hardware and isn't installable here).
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

import paho.mqtt.client as mqtt

from conftest import FIXTURES_DIR, TELEMETRY_MODULE_PATH


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


def _env_with_fake_jtop(scenario):
    env = dict(os.environ)
    fixture_dir = FIXTURES_DIR / f"fake_jtop_{scenario}"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(fixture_dir) + (os.pathsep + existing if existing else "")
    return env


@pytest.fixture
def telemetry_service_factory(mosquitto_broker):
    procs = []

    def start(topic="orin/status", jtop_scenario="success", extra_args=None):
        args = [
            sys.executable, str(TELEMETRY_MODULE_PATH),
            "--broker", "127.0.0.1",
            "--port", str(mosquitto_broker),
            "--topic", topic,
            "--interval", "1",
            "--loglevel", "DEBUG",
        ]
        if extra_args:
            args.extend(extra_args)
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=_env_with_fake_jtop(jtop_scenario),
        )
        svc = ServiceProcess(proc)
        procs.append(svc)
        return svc

    yield start

    for svc in procs:
        svc.stop()


@pytest.fixture
def probe_client(mosquitto_broker):
    received = []
    client = mqtt.Client(client_id="telemetry-probe", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
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
    retain=1: per the MQTT spec, a client that was *already* subscribed
    when a retained-flagged message is published receives it live, with
    retain=0 regardless of how the publisher set the flag - retain=1 is
    reserved for messages replayed to a client on subscribe. So "was
    this actually stored as retained" has to be checked from a fresh
    subscriber's point of view, not the original probe's.
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


def test_periodic_telemetry_is_published(telemetry_service_factory, probe_client):
    client, received = probe_client
    client.subscribe("orin/status", qos=1)
    time.sleep(0.3)

    svc = telemetry_service_factory(topic="orin/status")
    assert svc.wait_for_log("Telemetry service started"), svc.text()

    assert _wait_until(lambda: len(received) >= 1), svc.text()
    topic, payload, retain = received[-1]
    body = json.loads(payload)
    assert body["status"] == "online"
    assert body["cpu_avg"] is not None
    assert body["temp_cpu"] == 45.0
    assert retain is False


def test_lwt_fires_on_ungraceful_kill(telemetry_service_factory, probe_client, mosquitto_broker):
    client, received = probe_client
    client.subscribe("orin/status", qos=1)
    time.sleep(0.3)

    svc = telemetry_service_factory(topic="orin/status")
    assert svc.wait_for_log("Telemetry service started"), svc.text()
    assert _wait_until(lambda: len(received) >= 1), "expected at least one telemetry publish before kill"
    received.clear()

    svc.proc.kill()  # SIGKILL: no signal handler runs, only the broker's LWT can announce this
    svc.proc.wait(timeout=5)

    assert _wait_until(lambda: len(received) >= 1, timeout=10), "LWT was not published by the broker"
    topic, payload, retain = received[-1]
    assert json.loads(payload) == {"heartbeat": 0, "status": "offline"}
    _assert_retained(mosquitto_broker, "orin/status", {"heartbeat": 0, "status": "offline"})


def test_offline_status_on_clean_shutdown(telemetry_service_factory, probe_client, mosquitto_broker):
    client, received = probe_client
    client.subscribe("orin/status", qos=1)
    time.sleep(0.3)

    svc = telemetry_service_factory(topic="orin/status")
    assert svc.wait_for_log("Telemetry service started"), svc.text()
    assert _wait_until(lambda: len(received) >= 1), svc.text()
    received.clear()

    svc.proc.send_signal(signal.SIGTERM)
    rc = svc.proc.wait(timeout=10)
    assert rc == 0, svc.text()
    assert "Shutdown complete." in svc.text()

    assert _wait_until(lambda: len(received) >= 1), "no offline status observed after clean shutdown"
    topic, payload, retain = received[-1]
    assert json.loads(payload) == {"heartbeat": 0, "status": "offline"}
    _assert_retained(mosquitto_broker, "orin/status", {"heartbeat": 0, "status": "offline"})


def test_jtop_failure_exits_nonzero_and_still_publishes_offline(telemetry_service_factory, probe_client, mosquitto_broker):
    """Regression test for the systemd-restart fix, end to end: a jtop
    that fails to open must make the process exit non-zero (so
    Restart=on-failure fires) while still publishing the offline status
    for the MQTT connection it did manage to establish."""
    client, received = probe_client
    client.subscribe("orin/status", qos=1)
    time.sleep(0.3)

    svc = telemetry_service_factory(topic="orin/status", jtop_scenario="fail")
    rc = svc.proc.wait(timeout=10)

    assert rc != 0, "jtop failure should exit non-zero so systemd restarts the service: " + svc.text()
    assert "Error in main loop" in svc.text()
    assert "Shutdown complete." in svc.text()

    assert _wait_until(lambda: len(received) >= 1), "no offline status observed after jtop failure: " + svc.text()
    topic, payload, retain = received[-1]
    assert json.loads(payload) == {"heartbeat": 0, "status": "offline"}
    _assert_retained(mosquitto_broker, "orin/status", {"heartbeat": 0, "status": "offline"})
