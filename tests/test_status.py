"""
Tests for the periodic/event-driven MQTT status message: build_status_payload(),
build_queue_snapshot(), update_queue_progress(), the queue_status lifecycle
inside ollama_worker(), and the status_publisher() background thread.

Complements test_callbacks.py (on_connect/on_message's role in this - the
initial online announcement and the immediate publish on enqueue) and
test_shutdown.py (the offline payload's shape on LWT/clean shutdown).
"""
import argparse
import json
import socket
import threading
import time
from unittest.mock import MagicMock

import pytest
from conftest import load_mqtt_llm

# ------------------------------
# build_status_payload / build_queue_snapshot
# ------------------------------

def test_build_status_payload_offline_is_all_null(loaded_module):
    m = loaded_module
    assert m.build_status_payload(online=False) == {"status": "offline", "heartbeat": None, "queue": None}


def test_build_status_payload_online_with_empty_queue(loaded_module):
    m = loaded_module
    payload = m.build_status_payload(online=True)
    assert payload["status"] == "online"
    assert payload["queue"] == []
    assert isinstance(payload["heartbeat"], int)


def test_build_queue_snapshot_preserves_insertion_order(loaded_module):
    """The snapshot is the queue's only order record - insertion order into
    queue_status must match request_queue's own FIFO processing order."""
    m = loaded_module
    with m.queue_status_lock:
        m.queue_status["first"] = {"response_chars": -1, "thinking_chars": -1}
        m.queue_status["second"] = {"response_chars": 0, "thinking_chars": 0}
        m.queue_status["third"] = {"response_chars": 12, "thinking_chars": 3}

    assert m.build_queue_snapshot() == [
        {"request_id": "first", "response_chars": -1, "thinking_chars": -1},
        {"request_id": "second", "response_chars": 0, "thinking_chars": 0},
        {"request_id": "third", "response_chars": 12, "thinking_chars": 3},
    ]


def test_update_queue_progress_ignores_missing_entry(loaded_module):
    """A late progress update for a request already removed from
    queue_status (e.g. drained by shutdown) must be a silent no-op, not
    a KeyError - stream_ollama_generate has no way to know that happened."""
    m = loaded_module
    m.update_queue_progress("never-registered", response_chars=5, thinking_chars=1)  # must not raise
    assert "never-registered" not in m.queue_status


# ------------------------------
# --interval / STATUS_INTERVAL
# ------------------------------

def test_positive_int_accepts_positive_value(loaded_module):
    m = loaded_module
    assert m._positive_int("10") == 10


def test_positive_int_rejects_zero(loaded_module):
    m = loaded_module
    with pytest.raises(argparse.ArgumentTypeError):
        m._positive_int("0")


def test_positive_int_rejects_negative(loaded_module):
    m = loaded_module
    with pytest.raises(argparse.ArgumentTypeError):
        m._positive_int("-5")


def test_interval_arg_sets_status_interval():
    m = load_mqtt_llm(["--broker", "127.0.0.1", "--interval", "3"])
    assert m.STATUS_INTERVAL == 3


def test_interval_defaults_to_ten_seconds(loaded_module):
    m = loaded_module
    assert m.STATUS_INTERVAL == 10


# ------------------------------
# ollama_worker(): queue_status lifecycle
# ------------------------------

def _make_request(module, **overrides):
    fields = {"request_id": "w-1", "model": "llama3", "system": "sys", "user": "hello"}
    fields.update(overrides)
    return module.OllamaRequest(**fields)


def test_ollama_worker_sets_zero_on_dequeue_before_generation_starts(loaded_module, monkeypatch):
    """The -1 -> 0 transition ("queued" -> "started, nothing generated
    yet") must happen as soon as the worker picks the request up - before
    stream_ollama_generate() (and thus any real I/O) even begins."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    started = threading.Event()
    release = threading.Event()

    def fake_stream(request):
        started.set()
        assert release.wait(timeout=5), "test did not release fake_stream in time"
        return {"response": "ok"}

    monkeypatch.setattr(m, "stream_ollama_generate", fake_stream)

    request = _make_request(m, request_id="w-4")
    with m.queue_status_lock:
        m.queue_status["w-4"] = {"response_chars": -1, "thinking_chars": -1}
    m.request_queue.put(request)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        assert started.wait(timeout=5), "ollama_worker never reached stream_ollama_generate"
        assert m.queue_status["w-4"] == {"response_chars": 0, "thinking_chars": 0}
    finally:
        release.set()
        m.request_queue.put(m.SHUTDOWN_SENTINEL)
        t.join(timeout=5)

    assert not t.is_alive()
    assert "w-4" not in m.queue_status


def test_ollama_worker_removes_entry_after_successful_completion(loaded_module, fake_ollama, monkeypatch):
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    request = _make_request(m, request_id="w-1")
    with m.queue_status_lock:
        m.queue_status["w-1"] = {"response_chars": -1, "thinking_chars": -1}
    m.request_queue.put(request)
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert "w-1" not in m.queue_status


def test_ollama_worker_removes_entry_after_error(loaded_module, fake_ollama, monkeypatch):
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    fake_ollama.scenario = "http_error"
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    request = _make_request(m, request_id="w-err")
    with m.queue_status_lock:
        m.queue_status["w-err"] = {"response_chars": -1, "thinking_chars": -1}
    m.request_queue.put(request)
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert "w-err" not in m.queue_status


def test_ollama_worker_publishes_status_on_completion(loaded_module, fake_ollama, monkeypatch):
    m = loaded_module
    publish_mock = MagicMock(return_value=MagicMock(rc=0))
    monkeypatch.setattr(m.client, "publish", publish_mock)
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    request = _make_request(m, request_id="w-pub")
    with m.queue_status_lock:
        m.queue_status["w-pub"] = {"response_chars": -1, "thinking_chars": -1}
    m.request_queue.put(request)
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()

    status_calls = [c for c in publish_mock.call_args_list if c.args[0] == m.MQTT_STATUS_TOPIC]
    assert status_calls, "ollama_worker never published a status update on completion"
    body = json.loads(status_calls[-1].args[1])
    assert body["queue"] == []


def test_ollama_worker_skips_status_publish_when_shutdown_begins_mid_generation(loaded_module, monkeypatch):
    """The main shutdown block publishes its own final offline status right
    after draining the worker - an extra 'online' status from the worker's
    own completion handling, racing that final offline publish, would be
    redundant and misleading. Simulates shutdown_event being set while a
    generation is still in flight, then checks the completion handler's
    own guard skips the publish."""
    m = loaded_module
    publish_mock = MagicMock(return_value=MagicMock(rc=0))
    monkeypatch.setattr(m.client, "publish", publish_mock)

    def fake_stream(request):
        m.shutdown_event.set()  # shutdown arrives while this "generation" is in flight
        return {"response": "ok"}

    monkeypatch.setattr(m, "stream_ollama_generate", fake_stream)

    request = _make_request(m, request_id="w-shutdown")
    with m.queue_status_lock:
        m.queue_status["w-shutdown"] = {"response_chars": -1, "thinking_chars": -1}
    m.request_queue.put(request)
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        t.join(timeout=5)
        assert not t.is_alive()
        assert "w-shutdown" not in m.queue_status
        status_calls = [c for c in publish_mock.call_args_list if c.args[0] == m.MQTT_STATUS_TOPIC]
        assert not status_calls
    finally:
        m.shutdown_event.clear()


def test_ollama_worker_updates_progress_during_streaming(loaded_module, fake_ollama, monkeypatch):
    """Regression test for the per-chunk update: response_chars must
    reflect the actual running total mid-stream, not just the final
    value once the whole generation completes."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    fake_ollama.scenario = "hang"
    fake_ollama.hang_seconds = 30
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    request = _make_request(m, request_id="w-2")
    with m.queue_status_lock:
        m.queue_status["w-2"] = {"response_chars": -1, "thinking_chars": -1}
    m.request_queue.put(request)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not fake_ollama.received_requests:
            time.sleep(0.02)
        assert fake_ollama.received_requests, "request never reached the fake Ollama server"

        deadline = time.monotonic() + 2
        entry = m.queue_status.get("w-2")
        while time.monotonic() < deadline and entry is not None and entry["response_chars"] <= 0:
            time.sleep(0.02)
            entry = m.queue_status.get("w-2")

        assert entry is not None
        assert entry["response_chars"] == len("partial-")
        assert entry["thinking_chars"] == 0
    finally:
        if fake_ollama.last_connection is not None:
            try:
                fake_ollama.last_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        m.request_queue.put(m.SHUTDOWN_SENTINEL)
        t.join(timeout=10)

    assert not t.is_alive()


def test_ollama_worker_tracks_thinking_chars_independently_of_response_chars(loaded_module, fake_ollama, monkeypatch):
    """response_chars and thinking_chars must advance independently, so a
    reasoning model's "thinking" phase is separately observable from its
    final-response phase, not folded into one combined counter."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    fake_ollama.chunk_delay = 0.3  # slow enough to catch mid-stream state deterministically
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    request = _make_request(m, request_id="w-3")
    with m.queue_status_lock:
        m.queue_status["w-3"] = {"response_chars": -1, "thinking_chars": -1}
    m.request_queue.put(request)
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 3
        entry = m.queue_status.get("w-3")
        while time.monotonic() < deadline and entry is not None and entry["thinking_chars"] <= 0:
            time.sleep(0.02)
            entry = m.queue_status.get("w-3")

        assert entry is not None, "queue_status entry disappeared before thinking progress was observed"
        assert entry["thinking_chars"] == len("greeting")
        assert entry["response_chars"] == len("Hello, world!")
    finally:
        t.join(timeout=10)

    assert not t.is_alive()
    assert "w-3" not in m.queue_status


# ------------------------------
# status_publisher() background thread
# ------------------------------

def test_status_publisher_publishes_periodically_until_shutdown(loaded_module, monkeypatch):
    m = loaded_module
    m.STATUS_INTERVAL = 0.05
    publish_mock = MagicMock(return_value=MagicMock(rc=0))
    monkeypatch.setattr(m.client, "publish", publish_mock)

    t = threading.Thread(target=m.status_publisher, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and publish_mock.call_count < 3:
            time.sleep(0.01)
        assert publish_mock.call_count >= 3, f"only {publish_mock.call_count} status publishes observed"
        for call in publish_mock.call_args_list:
            assert call.args[0] == m.MQTT_STATUS_TOPIC
            assert json.loads(call.args[1])["status"] == "online"
    finally:
        m.shutdown_event.set()
        t.join(timeout=2)

    assert not t.is_alive()


def test_status_publisher_stops_promptly_on_shutdown(loaded_module, monkeypatch):
    m = loaded_module
    m.STATUS_INTERVAL = 10  # long enough that only the shutdown signal (not the timer) can end the loop in time
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))

    t = threading.Thread(target=m.status_publisher, daemon=True)
    t.start()
    try:
        time.sleep(0.1)
        assert t.is_alive()
        start = time.monotonic()
        m.shutdown_event.set()
        t.join(timeout=2)
        elapsed = time.monotonic() - start

        assert not t.is_alive()
        assert elapsed < 1, f"status_publisher took {elapsed:.2f}s to notice shutdown_event"
    finally:
        m.shutdown_event.clear()
