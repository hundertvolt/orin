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


def _msg(payload: bytes, topic="orin/ollama/request"):
    msg = MagicMock()
    msg.payload = payload
    msg.topic = topic
    return msg

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
    queue_status (keyed by the internal seq counter, not request_id - see
    the "queue tracking identity" tests below for why) must match
    request_queue's own FIFO processing order."""
    m = loaded_module
    with m.queue_status_lock:
        m.queue_status[1] = {"request_id": "first", "response_chars": -1, "thinking_chars": -1}
        m.queue_status[2] = {"request_id": "second", "response_chars": 0, "thinking_chars": 0}
        m.queue_status[3] = {"request_id": "third", "response_chars": 12, "thinking_chars": 3}

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
    m.update_queue_progress(999999, response_chars=5, thinking_chars=1)  # must not raise
    assert 999999 not in m.queue_status


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


def _enqueue(module, request_id, seq=None, **overrides):
    """Register a queue_status entry and push (seq, request) onto
    request_queue, mirroring what on_message does for a real MQTT
    request - but with direct control over `seq`, so tests can exercise
    ollama_worker() without going through the MQTT callback plumbing.

    request_id is deliberately NOT used as the queue_status key (a
    caller-supplied request_id may repeat, or be empty) - `seq` is an
    internal, always-unique tracking key, auto-assigned from the same
    counter production code uses unless the test passes its own."""
    request = _make_request(module, request_id=request_id, **overrides)
    if seq is None:
        seq = next(module._queue_seq_counter)
    with module.queue_status_lock:
        module.queue_status[seq] = {"request_id": request_id, "response_chars": -1, "thinking_chars": -1}
    module.request_queue.put((seq, request))
    return seq


def _pending_request_ids(module):
    return [entry["request_id"] for entry in module.build_queue_snapshot()]


def test_ollama_worker_sets_zero_on_dequeue_before_generation_starts(loaded_module, monkeypatch):
    """The -1 -> 0 transition ("queued" -> "started, nothing generated
    yet") must happen as soon as the worker picks the request up - before
    stream_ollama_generate() (and thus any real I/O) even begins.

    thinking_chars must NOT follow response_chars into 0 here: whether
    this request's model produces any "thinking" output at all isn't
    known until (if ever) stream_ollama_generate actually observes a
    "thinking" key in a chunk - regression test for a reported bug where
    thinking_chars incorrectly jumped to 0 for a non-thinking model."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    started = threading.Event()
    release = threading.Event()

    def fake_stream(request, seq=None):
        started.set()
        assert release.wait(timeout=5), "test did not release fake_stream in time"
        return {"response": "ok"}

    monkeypatch.setattr(m, "stream_ollama_generate", fake_stream)

    seq = _enqueue(m, "w-4")

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        assert started.wait(timeout=5), "ollama_worker never reached stream_ollama_generate"
        assert m.queue_status[seq] == {"request_id": "w-4", "response_chars": 0, "thinking_chars": -1}
    finally:
        release.set()
        m.request_queue.put(m.SHUTDOWN_SENTINEL)
        t.join(timeout=5)

    assert not t.is_alive()
    assert seq not in m.queue_status


def test_ollama_worker_removes_entry_after_successful_completion(loaded_module, fake_ollama, monkeypatch):
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    seq = _enqueue(m, "w-1")
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert seq not in m.queue_status


def test_ollama_worker_removes_entry_after_error(loaded_module, fake_ollama, monkeypatch):
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    fake_ollama.scenario = "http_error"
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    seq = _enqueue(m, "w-err")
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert seq not in m.queue_status


def test_ollama_worker_publishes_status_on_completion(loaded_module, fake_ollama, monkeypatch):
    m = loaded_module
    publish_mock = MagicMock(return_value=MagicMock(rc=0))
    monkeypatch.setattr(m.client, "publish", publish_mock)
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    _enqueue(m, "w-pub")
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

    def fake_stream(request, seq=None):
        m.shutdown_event.set()  # shutdown arrives while this "generation" is in flight
        return {"response": "ok"}

    monkeypatch.setattr(m, "stream_ollama_generate", fake_stream)

    seq = _enqueue(m, "w-shutdown")
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        t.join(timeout=5)
        assert not t.is_alive()
        assert seq not in m.queue_status
        status_calls = [c for c in publish_mock.call_args_list if c.args[0] == m.MQTT_STATUS_TOPIC]
        assert not status_calls
    finally:
        m.shutdown_event.clear()


def test_ollama_worker_updates_progress_during_streaming(loaded_module, fake_ollama, monkeypatch):
    """Regression test for the per-chunk update: response_chars must
    reflect the actual running total mid-stream, not just the final
    value once the whole generation completes. This scenario's chunks
    never include a "thinking" key at all (a non-thinking model), so
    thinking_chars must stay at -1 throughout, never touched."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    fake_ollama.scenario = "hang"
    fake_ollama.hang_seconds = 30
    fake_ollama.chunks = [{"response": "partial-"}]  # no "thinking" key: a non-thinking model
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    seq = _enqueue(m, "w-2")

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not fake_ollama.received_requests:
            time.sleep(0.02)
        assert fake_ollama.received_requests, "request never reached the fake Ollama server"

        deadline = time.monotonic() + 2
        entry = m.queue_status.get(seq)
        while time.monotonic() < deadline and entry is not None and entry["response_chars"] <= 0:
            time.sleep(0.02)
            entry = m.queue_status.get(seq)

        assert entry is not None
        assert entry["response_chars"] == len("partial-")
        assert entry["thinking_chars"] == -1
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

    seq = _enqueue(m, "w-3")
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 3
        entry = m.queue_status.get(seq)
        while time.monotonic() < deadline and entry is not None and entry["thinking_chars"] <= 0:
            time.sleep(0.02)
            entry = m.queue_status.get(seq)

        assert entry is not None, "queue_status entry disappeared before thinking progress was observed"
        assert entry["thinking_chars"] == len("greeting")
        assert entry["response_chars"] == len("Hello, world!")
    finally:
        t.join(timeout=10)

    assert not t.is_alive()
    assert seq not in m.queue_status


def test_thinking_chars_reaches_zero_when_key_present_but_content_empty(loaded_module, fake_ollama, monkeypatch):
    """A model can signal "thinking has started" via an empty "thinking"
    string before it has any actual content - that must read as 0
    ("started, nothing generated yet"), distinct from -1 ("no thinking
    activity observed at all"), and distinct from a model that never
    sends the key at all."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    fake_ollama.scenario = "hang"
    fake_ollama.hang_seconds = 30
    fake_ollama.chunks = [{"response": "", "thinking": ""}]  # key present, but nothing generated yet
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    seq = _enqueue(m, "w-empty-thinking")

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        entry = m.queue_status.get(seq)
        while time.monotonic() < deadline and entry is not None and entry["thinking_chars"] == -1:
            time.sleep(0.02)
            entry = m.queue_status.get(seq)

        assert entry is not None
        assert entry["thinking_chars"] == 0, "an observed-but-empty thinking key must read as 0, not -1"
        assert entry["response_chars"] == 0
    finally:
        if fake_ollama.last_connection is not None:
            try:
                fake_ollama.last_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        m.request_queue.put(m.SHUTDOWN_SENTINEL)
        t.join(timeout=10)

    assert not t.is_alive()


# ------------------------------
# queue length/order across a request's lifecycle
#
# Regression coverage for a reported bug: the "queue" snapshot only ever
# showed a single element, and went empty as soon as that one finished
# even with further requests still genuinely pending behind it.
# ------------------------------

def test_queue_snapshot_starts_empty_on_fresh_service_start(loaded_module):
    m = loaded_module
    assert m.build_queue_snapshot() == []


def test_queue_snapshot_shows_all_pending_requests_before_worker_touches_them(loaded_module):
    """A burst of arrivals landing before the worker has processed any of
    them (a "populated queue") must all be visible at once, in order."""
    m = loaded_module
    for rid in ("a", "b", "c"):
        _enqueue(m, rid)

    assert m.build_queue_snapshot() == [
        {"request_id": "a", "response_chars": -1, "thinking_chars": -1},
        {"request_id": "b", "response_chars": -1, "thinking_chars": -1},
        {"request_id": "c", "response_chars": -1, "thinking_chars": -1},
    ]


def test_queue_shrinks_one_at_a_time_while_later_entries_stay_pending(loaded_module, monkeypatch):
    """Core regression test: process a 3-item backlog one at a time and
    check the snapshot at every step - each completion must remove only
    the head entry, never the whole list, and the next entry must become
    the new head (0/-1) while later ones remain untouched at -1/-1."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))

    gates = {rid: threading.Event() for rid in ("a", "b", "c")}
    started = {rid: threading.Event() for rid in ("a", "b", "c")}

    def fake_stream(request, seq=None):
        started[request.request_id].set()
        assert gates[request.request_id].wait(timeout=5), f"{request.request_id} never released"
        return {"response": "ok"}

    monkeypatch.setattr(m, "stream_ollama_generate", fake_stream)

    seqs = {rid: _enqueue(m, rid) for rid in ("a", "b", "c")}
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        assert started["a"].wait(timeout=5)
        snapshot = m.build_queue_snapshot()
        assert [e["request_id"] for e in snapshot] == ["a", "b", "c"]
        assert snapshot[0] == {"request_id": "a", "response_chars": 0, "thinking_chars": -1}
        assert snapshot[1] == {"request_id": "b", "response_chars": -1, "thinking_chars": -1}
        assert snapshot[2] == {"request_id": "c", "response_chars": -1, "thinking_chars": -1}

        gates["a"].set()
        assert started["b"].wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and seqs["a"] in m.queue_status:
            time.sleep(0.01)
        assert seqs["a"] not in m.queue_status

        snapshot = m.build_queue_snapshot()
        assert [e["request_id"] for e in snapshot] == ["b", "c"], \
            "finishing the head must not empty the whole queue while 'c' is still pending"
        assert snapshot[0] == {"request_id": "b", "response_chars": 0, "thinking_chars": -1}
        assert snapshot[1] == {"request_id": "c", "response_chars": -1, "thinking_chars": -1}

        gates["b"].set()
        assert started["c"].wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and seqs["b"] in m.queue_status:
            time.sleep(0.01)
        assert seqs["b"] not in m.queue_status

        snapshot = m.build_queue_snapshot()
        assert [e["request_id"] for e in snapshot] == ["c"]
        assert snapshot[0] == {"request_id": "c", "response_chars": 0, "thinking_chars": -1}

        gates["c"].set()
        t.join(timeout=5)
        assert not t.is_alive()
        assert m.build_queue_snapshot() == []
    finally:
        for g in gates.values():
            g.set()
        if t.is_alive():
            t.join(timeout=5)


def test_queue_reflects_new_arrival_added_to_an_already_in_flight_queue(loaded_module, monkeypatch):
    """A request added while others are already queued/processing must
    appear at the tail immediately, without disturbing the existing
    entries' order or values."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))

    gates = {rid: threading.Event() for rid in ("a", "b")}
    started = {rid: threading.Event() for rid in ("a", "b")}

    def fake_stream(request, seq=None):
        started[request.request_id].set()
        assert gates[request.request_id].wait(timeout=5)
        return {"response": "ok"}

    monkeypatch.setattr(m, "stream_ollama_generate", fake_stream)

    for rid in ("a", "b"):
        _enqueue(m, rid)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        assert started["a"].wait(timeout=5)
        assert [e["request_id"] for e in m.build_queue_snapshot()] == ["a", "b"]

        # New arrival "c" lands on top of the already in-flight "a"/"b".
        _enqueue(m, "c")

        snapshot = m.build_queue_snapshot()
        assert [e["request_id"] for e in snapshot] == ["a", "b", "c"], \
            "a live arrival must extend the queue, not replace or hide existing entries"
        assert snapshot[1] == {"request_id": "b", "response_chars": -1, "thinking_chars": -1}
        assert snapshot[2] == {"request_id": "c", "response_chars": -1, "thinking_chars": -1}
    finally:
        for g in gates.values():
            g.set()
        m.request_queue.put(m.SHUTDOWN_SENTINEL)
        t.join(timeout=5)

    assert not t.is_alive()


def test_queue_resets_cleanly_after_draining_then_new_arrival(loaded_module, monkeypatch):
    """After the queue drains back to empty, a brand new arrival must show
    up alone - no stale leftover entry from the previous, already-finished
    batch (covers "starting from an existing queue which ran empty")."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    gate = threading.Event()
    started = threading.Event()

    def fake_stream(request, seq=None):
        started.set()
        assert gate.wait(timeout=5)
        return {"response": "ok"}

    monkeypatch.setattr(m, "stream_ollama_generate", fake_stream)

    _enqueue(m, "first")

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        assert started.wait(timeout=5)
        gate.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and m.build_queue_snapshot() != []:
            time.sleep(0.01)
        assert m.build_queue_snapshot() == [], "queue never genuinely drained to empty"

        started.clear()
        gate.clear()
        _enqueue(m, "second")

        assert started.wait(timeout=5)
        assert m.build_queue_snapshot() == [
            {"request_id": "second", "response_chars": 0, "thinking_chars": -1},
        ], "a fresh arrival after the queue emptied must not carry over stale entries"
        gate.set()
    finally:
        gate.set()
        m.request_queue.put(m.SHUTDOWN_SENTINEL)
        t.join(timeout=5)

    assert not t.is_alive()


def test_short_request_processes_promptly_behind_a_long_running_head(loaded_module, fake_ollama, monkeypatch):
    """A short (fast) request queued behind a long-running (hanging) one
    must remain visible at -1/-1 the entire time the head is streaming,
    and must actually complete promptly once its turn comes - it must not
    be starved or dropped by whatever kept the long one visible."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))
    fake_ollama.scenario = "hang"
    fake_ollama.hang_seconds = 30
    fake_ollama.chunks = [{"response": "partial-"}]
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    seq_long = _enqueue(m, "long")
    seq_short = _enqueue(m, "short")

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not fake_ollama.received_requests:
            time.sleep(0.02)
        assert fake_ollama.received_requests, "the long request never reached the fake Ollama server"

        deadline = time.monotonic() + 2
        snapshot = m.build_queue_snapshot()
        while time.monotonic() < deadline and (not snapshot or snapshot[0]["response_chars"] <= 0):
            time.sleep(0.02)
            snapshot = m.build_queue_snapshot()

        assert [e["request_id"] for e in snapshot] == ["long", "short"]
        assert snapshot[0]["response_chars"] > 0  # "long" is actively streaming
        assert snapshot[1] == {"request_id": "short", "response_chars": -1, "thinking_chars": -1}

        # Force-close "long"'s connection so the fake server can move on to
        # "short" - give "short" a real "done" marker so the client-side
        # loop breaks on its own instead of blocking on the (still
        # keep-alive) connection for more data that never arrives.
        fake_ollama.scenario = "success"
        fake_ollama.chunks = [{"response": "short-response", "done": True, "done_reason": "stop"}]
        if fake_ollama.last_connection is not None:
            try:
                fake_ollama.last_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and seq_long in m.queue_status:
            time.sleep(0.02)
        assert seq_long not in m.queue_status

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and seq_short in m.queue_status:
            time.sleep(0.02)
        assert seq_short not in m.queue_status, "the short request behind the long one never completed"
    finally:
        m.request_queue.put(m.SHUTDOWN_SENTINEL)
        t.join(timeout=10)

    assert not t.is_alive()


# ------------------------------
# request_id is an arbitrary, caller-supplied label - never an identity
#
# Root cause of the reported "queue only ever shows one element" bug:
# it was keyed by request_id, so multiple requests sharing one request_id
# (an arbitrary string the client can reuse, or even leave "") collapsed
# onto a single queue_status entry. queue_status is now keyed by an
# internal seq counter instead; these tests hold that contract directly,
# independent of what request_id content a client happens to send.
# ------------------------------

def test_duplicate_request_ids_get_independent_queue_entries(loaded_module):
    """Exact regression test for the reported bug: three requests all
    using the SAME request_id must still show up as three distinct queue
    entries, each independently trackable - not collapse into one."""
    m = loaded_module
    for _ in range(3):
        _enqueue(m, "same-id")

    snapshot = m.build_queue_snapshot()
    assert len(snapshot) == 3, f"duplicate request_ids must not collapse into fewer queue entries: {snapshot}"
    assert all(e["request_id"] == "same-id" for e in snapshot)
    assert all(e == {"request_id": "same-id", "response_chars": -1, "thinking_chars": -1} for e in snapshot)


def test_duplicate_request_id_finishing_one_leaves_the_others_pending(loaded_module, monkeypatch):
    """Processing one of several same-ID requests to completion must only
    remove *that* queue slot - the others sharing the same request_id
    (and not yet started) must remain fully visible and independently
    trackable, matching the reported bug's exact symptom ("queue field
    stays empty" despite more of the same-ID batch still pending)."""
    m = loaded_module
    monkeypatch.setattr(m.client, "publish", MagicMock(return_value=MagicMock(rc=0)))

    gates = [threading.Event() for _ in range(3)]
    started = [threading.Event() for _ in range(3)]
    call_index = {"n": 0}

    def fake_stream(request, seq=None):
        i = call_index["n"]
        call_index["n"] += 1
        started[i].set()
        assert gates[i].wait(timeout=5)
        return {"response": "ok"}

    monkeypatch.setattr(m, "stream_ollama_generate", fake_stream)

    seqs = [_enqueue(m, "same-id") for _ in range(3)]
    m.request_queue.put(m.SHUTDOWN_SENTINEL)

    t = threading.Thread(target=m.ollama_worker, daemon=True)
    t.start()
    try:
        assert started[0].wait(timeout=5)
        assert len(m.build_queue_snapshot()) == 3

        gates[0].set()  # finish the first "same-id" request
        assert started[1].wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and seqs[0] in m.queue_status:
            time.sleep(0.01)
        assert seqs[0] not in m.queue_status

        snapshot = m.build_queue_snapshot()
        assert len(snapshot) == 2, \
            f"finishing one same-ID request must not empty the queue while 2 more are pending: {snapshot}"
        assert all(e["request_id"] == "same-id" for e in snapshot)

        gates[1].set()
        assert started[2].wait(timeout=5)
        gates[2].set()
        t.join(timeout=5)
        assert not t.is_alive()
        assert m.build_queue_snapshot() == []
    finally:
        for g in gates:
            g.set()
        if t.is_alive():
            t.join(timeout=5)


def test_empty_string_request_id_is_a_valid_independent_queue_entry(loaded_module):
    """An empty request_id is a legal value (OllamaRequest only requires
    it to be a string) and must be handled exactly like any other label -
    including allowing several empty-ID requests to coexist as distinct
    queue entries."""
    m = loaded_module
    _enqueue(m, "")
    _enqueue(m, "")

    snapshot = m.build_queue_snapshot()
    assert len(snapshot) == 2
    assert all(e["request_id"] == "" for e in snapshot)


# request_id: str = Field(strict=True) is pydantic's *entire* contract -
# strict=True only blocks type coercion (e.g. an int or bool being
# silently turned into a string); it imposes no length limit, no
# character/charset restriction, and no uniqueness requirement. Any of
# the following is therefore a perfectly valid, pydantic-accepted
# request_id, and the queue/status code must not assume anything more
# about it than "it's a str" - no implicit "reasonable length",
# "printable ASCII", "no duplicates", etc.
WEIRD_BUT_VALID_REQUEST_IDS = [
    "",
    " ",
    "\t\n\r",
    "a" * 10_000,
    "unicode-日本語-🚀-emoji",
    'has "quotes" and \\ backslashes and \n a newline',
    "null-byte-\x00-embedded",
    "123",  # numeric-looking, but must stay a plain string, never coerced/used as an index
    "same-id",
]


@pytest.mark.parametrize("weird_id", WEIRD_BUT_VALID_REQUEST_IDS)
def test_queue_snapshot_round_trips_arbitrary_request_id_content(loaded_module, weird_id):
    """The queue snapshot must carry any pydantic-valid request_id through
    unchanged, and the resulting status payload must still be valid,
    round-trippable JSON - json.dumps/json.loads must not choke on any of
    this content, and no character in it may corrupt the JSON structure."""
    m = loaded_module
    _enqueue(m, weird_id)

    snapshot = m.build_queue_snapshot()
    assert snapshot == [{"request_id": weird_id, "response_chars": -1, "thinking_chars": -1}]

    payload = m.build_status_payload(online=True)
    encoded = json.dumps(payload)  # must not raise
    decoded = json.loads(encoded)  # must round-trip byte-for-byte
    assert decoded["queue"] == [{"request_id": weird_id, "response_chars": -1, "thinking_chars": -1}]


@pytest.mark.parametrize("weird_id", WEIRD_BUT_VALID_REQUEST_IDS)
def test_on_message_accepts_and_tracks_arbitrary_request_id_content(loaded_module, weird_id):
    """Same coverage as above, but through the real on_message() entry
    point - pydantic validates weird_id as an ordinary str (it has no
    reason to reject any of these), and everything downstream (queueing,
    tracking, the immediate status publish) must handle it the same as
    any "normal-looking" request_id."""
    m = loaded_module
    cli = MagicMock()
    payload = json.dumps({
        "request_id": weird_id, "model": "llama3", "system": "s", "user": "u",
    }).encode("utf-8")

    m.on_message(cli, None, _msg(payload))

    assert m.request_queue.qsize() == 1
    _seq, queued = m.request_queue.get_nowait()
    assert queued.request_id == weird_id  # pydantic passed the content through unchanged

    assert m.build_queue_snapshot() == [{"request_id": weird_id, "response_chars": -1, "thinking_chars": -1}]

    cli.publish.assert_called_once()
    body = json.loads(cli.publish.call_args.args[1])  # must be valid JSON despite weird_id's content
    assert body["queue"] == [{"request_id": weird_id, "response_chars": -1, "thinking_chars": -1}]


def test_queue_tracks_a_mixed_batch_of_arbitrary_request_id_content_independently(loaded_module):
    """Several requests with wildly different (but each individually
    valid) request_id content, arriving together, must all be tracked as
    independent entries - odd content in one entry must not affect, merge
    with, or otherwise interfere with any other."""
    m = loaded_module
    for weird_id in WEIRD_BUT_VALID_REQUEST_IDS:
        _enqueue(m, weird_id)

    snapshot = m.build_queue_snapshot()
    assert len(snapshot) == len(WEIRD_BUT_VALID_REQUEST_IDS)
    assert [e["request_id"] for e in snapshot] == WEIRD_BUT_VALID_REQUEST_IDS
    assert all(e["response_chars"] == -1 and e["thinking_chars"] == -1 for e in snapshot)

    # Must still be valid, round-trippable JSON as a whole batch.
    encoded = json.dumps(m.build_status_payload(online=True))
    assert json.loads(encoded)["queue"] == snapshot


def test_on_message_accepts_request_with_all_empty_string_fields(loaded_module):
    """request_id, model, system, and user are all just `str` to
    pydantic - nothing requires any of them to be non-empty. A request
    that's entirely empty strings must still be validated, queued, and
    tracked without error."""
    m = loaded_module
    cli = MagicMock()
    payload = json.dumps({
        "request_id": "", "model": "", "system": "", "user": "",
    }).encode("utf-8")

    m.on_message(cli, None, _msg(payload))  # must not raise

    assert m.request_queue.qsize() == 1
    _seq, queued = m.request_queue.get_nowait()
    assert queued.model == "" and queued.system == "" and queued.user == ""
    assert m.build_queue_snapshot() == [{"request_id": "", "response_chars": -1, "thinking_chars": -1}]


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
