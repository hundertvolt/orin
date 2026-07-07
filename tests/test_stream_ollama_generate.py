"""
Tests for stream_ollama_generate() against a real local HTTP server that
mimics Ollama's streaming NDJSON protocol (real sockets, real
http.client parsing - only the actual Ollama/model backend is faked,
since we can't run one here).

Focus of the new coverage: the guarded conn.close() in the finally
block (previously unguarded - an OSError there would have masked
whatever exception was actually in flight, or crashed a clean return).
"""
import http.client
import socket
import threading
import time

import pytest


def _make_request(module, **overrides):
    fields = {
        "request_id": "req-1",
        "model": "llama3",
        "system": "sys",
        "user": "hello",
    }
    fields.update(overrides)
    return module.OllamaRequest(**fields)


def test_successful_stream_concatenates_response_and_thinking(loaded_module, fake_ollama):
    m = loaded_module
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    result = m.stream_ollama_generate(_make_request(m))

    assert result["response"] == "Hello, world!"
    assert result["thinking"] == "greeting"
    assert result["done"] is True
    assert result["done_reason"] == "stop"
    assert result["total_duration"] == 123
    # request/response body sanity: our OllamaRequest.to_ollama_options() was empty
    sent = fake_ollama.received_requests[0]
    assert sent["model"] == "llama3"
    assert sent["system"] == "sys"
    assert sent["prompt"] == "hello"
    assert "options" not in sent


def test_active_responses_cleaned_up_after_success(loaded_module, fake_ollama):
    m = loaded_module
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    m.stream_ollama_generate(_make_request(m))

    assert m.active_responses == {}


def test_options_included_when_set(loaded_module, fake_ollama):
    m = loaded_module
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    m.stream_ollama_generate(_make_request(m, temperature=0.2, top_k=10))

    sent = fake_ollama.received_requests[0]
    assert sent["options"] == {"temperature": 0.2, "top_k": 10}


def test_http_error_status_raises_runtime_error(loaded_module, fake_ollama):
    m = loaded_module
    fake_ollama.scenario = "http_error"
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    with pytest.raises(RuntimeError, match="Ollama returned HTTP 500"):
        m.stream_ollama_generate(_make_request(m))

    # cleanup still ran despite the error path
    assert m.active_responses == {}


def test_connection_refused_propagates_and_cleans_up(loaded_module):
    m = loaded_module
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = 1  # nothing listens on port 1

    with pytest.raises(OSError):
        m.stream_ollama_generate(_make_request(m))

    assert m.active_responses == {}


def test_shutdown_in_progress_converts_exception_to_cancelled(loaded_module):
    """When the connection breaks *because* we're shutting down, the
    caller (ollama_worker) needs a distinct signal so it logs a clean
    cancellation instead of an OLLAMA_ERROR."""
    m = loaded_module
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = 1  # connection error, but framed as a shutdown now
    m.shutdown_event.set()

    with pytest.raises(m.GenerationCancelled):
        m.stream_ollama_generate(_make_request(m))


def test_conn_close_oserror_during_cleanup_is_suppressed(loaded_module, fake_ollama, monkeypatch):
    """Direct regression test for the fix: conn.close() raising OSError
    in the finally block must not mask the real outcome or crash.

    Our NDJSON streaming responses have no Content-Length, so
    http.client's own getresponse() already closes the connection
    internally right after reading headers (it has no other way to
    know where the body ends). Our own finally-block close() is
    therefore *always* a second close on an already-closed connection
    in practice - exactly the "OSError: already closed" scenario this
    guard exists for. We simulate that here by letting the first
    (real, internal) close succeed and only failing subsequent ones.
    """
    m = loaded_module
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    real_close = http.client.HTTPConnection.close
    call_count = {"n": 0}

    def flaky_close(self):
        call_count["n"] += 1
        real_close(self)
        if call_count["n"] > 1:
            raise OSError("already shut down")

    monkeypatch.setattr(http.client.HTTPConnection, "close", flaky_close)

    result = m.stream_ollama_generate(_make_request(m))  # must not raise

    assert result["response"] == "Hello, world!"
    assert call_count["n"] >= 2, "expected both http.client's internal close and our own"
    assert m.active_responses == {}


def test_conn_close_oserror_does_not_hide_original_exception(loaded_module, monkeypatch):
    """If both the request *and* the cleanup close() fail, the original
    (more useful) exception must be what propagates, not the close error."""
    m = loaded_module
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = 1  # connection refused -> OSError from conn.connect()

    def flaky_close(self):
        raise OSError("close also failing")

    monkeypatch.setattr(http.client.HTTPConnection, "close", flaky_close)

    with pytest.raises(OSError):
        m.stream_ollama_generate(_make_request(m))


def test_registered_in_active_responses_while_in_flight(loaded_module, fake_ollama):
    """While a generation is streaming, its connection must be reachable
    via active_responses under the request_id key, and removed again
    once the call completes (successfully or not)."""
    m = loaded_module
    fake_ollama.scenario = "hang"
    fake_ollama.hang_seconds = 30  # longer than this test; we force-close instead of waiting it out
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    result_holder = {}

    def run():
        try:
            result_holder["value"] = m.stream_ollama_generate(_make_request(m, request_id="hang-1"))
        except Exception as e:
            result_holder["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not fake_ollama.received_requests:
            time.sleep(0.02)
        assert fake_ollama.received_requests, "request never reached the fake Ollama server"
        assert "hang-1" in m.active_responses
    finally:
        # The server never finishes this stream (no "done" chunk), so force
        # the connection closed to end the test deterministically instead of
        # waiting out OLLAMA_STREAM_IDLE_TIMEOUT.
        if fake_ollama.last_connection is not None:
            try:
                fake_ollama.last_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        t.join(timeout=10)

    assert not t.is_alive()
    # Notable (pre-existing, not introduced by our changes) behavior: a
    # connection closed mid-stream with no "done" chunk is NOT treated as
    # an error - `for line in resp` just sees EOF and stops. The caller
    # gets a clean-looking partial result instead of an exception.
    assert "value" in result_holder
    assert result_holder["value"]["response"] == "partial-"
    assert m.active_responses == {}


def test_conn_sock_is_none_once_streaming_has_started(loaded_module, fake_ollama):
    """Documents why the finally-block close() needed guarding, and why the
    shutdown block's `if conn.sock is not None: conn.sock.shutdown(...)`
    check matters: our NDJSON responses have no Content-Length, so
    http.client's getresponse() itself closes and nulls conn.sock right
    after reading headers (it has no other way to know where the body
    ends) - well before the body/streaming even starts. A shutdown-time
    unblock via conn.sock therefore cannot rely on conn.sock being live;
    the production code's None-check is what keeps that path safe."""
    m = loaded_module
    fake_ollama.scenario = "hang"
    fake_ollama.hang_seconds = 5
    m.OLLAMA_HOST = "127.0.0.1"
    m.OLLAMA_PORT = fake_ollama.server_address[1]

    t = threading.Thread(
        target=m.stream_ollama_generate, args=(_make_request(m, request_id="hang-2"),), daemon=True
    )
    t.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not fake_ollama.received_requests:
            time.sleep(0.02)
        assert fake_ollama.received_requests

        deadline = time.monotonic() + 2
        conn = m.active_responses.get("hang-2")
        while time.monotonic() < deadline and conn is not None and conn.sock is not None:
            time.sleep(0.02)
            conn = m.active_responses.get("hang-2")

        assert conn is not None
        assert conn.sock is None
    finally:
        # conn.sock is already None, so we can't unblock the read the way
        # the shutdown block would; force it from the server side instead
        # so this test doesn't leave a thread blocked for the full
        # OLLAMA_STREAM_IDLE_TIMEOUT (120s default) after the test ends.
        m.shutdown_event.set()
        if fake_ollama.last_connection is not None:
            try:
                fake_ollama.last_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            t.join(timeout=10)
        finally:
            m.shutdown_event.clear()
