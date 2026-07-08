"""
Unit tests for on_connect / on_disconnect / on_message.

These are the three paho-mqtt callbacks. paho re-raises any exception
that escapes a callback (suppress_exceptions=False by default), and
none of _loop() / loop_forever() / _thread_main() catch it - an
uncaught exception here silently kills the network loop thread
forever while the process keeps running. The core regression coverage
here is: no matter what breaks inside these callbacks, the function
call itself must always return normally.
"""
import json
from unittest.mock import MagicMock

import pytest

# ------------------------------
# on_connect
# ------------------------------

def test_on_connect_success_subscribes_and_announces_online(loaded_module):
    m = loaded_module
    cli = MagicMock()

    m.on_connect(cli, None, {}, 0, None)

    cli.subscribe.assert_called_once_with(f"{m.MQTT_TOPIC}/request", qos=1)
    cli.publish.assert_called_once_with(
        m.MQTT_STATUS_TOPIC, json.dumps({"status": "online"}), qos=1, retain=True
    )


def test_on_connect_failure_does_not_subscribe_or_publish(loaded_module):
    m = loaded_module
    cli = MagicMock()

    m.on_connect(cli, None, {}, 5, None)  # 5 = not authorized, e.g.

    cli.subscribe.assert_not_called()
    cli.publish.assert_not_called()


def test_on_connect_swallows_subscribe_exception(loaded_module, caplog):
    """Regression test for the paho-thread-killing bug: an exception from
    cli.subscribe() (e.g. a transient socket error) must not escape."""
    m = loaded_module
    cli = MagicMock()
    cli.subscribe.side_effect = RuntimeError("boom")

    m.on_connect(cli, None, {}, 0, None)  # must not raise

    assert "Unhandled error in on_connect" in caplog.text


def test_on_connect_swallows_publish_exception(loaded_module, caplog):
    m = loaded_module
    cli = MagicMock()
    cli.publish.side_effect = OSError("no route to host")

    m.on_connect(cli, None, {}, 0, None)  # must not raise

    assert "Unhandled error in on_connect" in caplog.text


# ------------------------------
# on_disconnect
# ------------------------------

def test_on_disconnect_clean(loaded_module, caplog):
    m = loaded_module
    caplog.set_level("INFO")
    m.on_disconnect(MagicMock(), None, {}, 0, None)
    assert "MQTT disconnected cleanly" in caplog.text


def test_on_disconnect_unexpected(loaded_module, caplog):
    m = loaded_module
    m.on_disconnect(MagicMock(), None, {}, 7, None)
    assert "Unexpected MQTT disconnection" in caplog.text


def test_on_disconnect_swallows_any_internal_exception(loaded_module, caplog, monkeypatch):
    """Even a failure inside our own logging call must not escape - this
    callback runs on paho's network thread same as on_connect/on_message."""
    m = loaded_module
    monkeypatch.setattr(m.log, "warning", MagicMock(side_effect=RuntimeError("log backend down")))

    m.on_disconnect(MagicMock(), None, {}, 7, None)  # must not raise


# ------------------------------
# on_message
# ------------------------------

def _msg(payload: bytes, topic="orin/ollama/request"):
    msg = MagicMock()
    msg.payload = payload
    msg.topic = topic
    return msg


def test_on_message_valid_request_is_queued(loaded_module):
    m = loaded_module
    payload = json.dumps({
        "request_id": "abc-123",
        "model": "llama3",
        "system": "You are helpful.",
        "user": "Hello",
    }).encode("utf-8")

    m.on_message(MagicMock(), None, _msg(payload))

    assert m.request_queue.qsize() == 1
    queued = m.request_queue.get_nowait()
    assert isinstance(queued, m.OllamaRequest)
    assert queued.request_id == "abc-123"
    assert queued.model == "llama3"


def test_on_message_optional_fields_pass_through(loaded_module):
    m = loaded_module
    payload = json.dumps({
        "request_id": "abc-124",
        "model": "llama3",
        "system": "sys",
        "user": "hi",
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
    }).encode("utf-8")

    m.on_message(MagicMock(), None, _msg(payload))

    queued = m.request_queue.get_nowait()
    assert queued.to_ollama_options() == {"temperature": 0.5, "top_p": 0.9, "top_k": 40}


def test_on_message_invalid_utf8_is_handled(loaded_module, caplog):
    m = loaded_module
    m.on_message(MagicMock(), None, _msg(b"\xff\xfe\xfa not utf8"))

    assert m.request_queue.qsize() == 0
    assert "Failed to decode message payload" in caplog.text


def test_on_message_invalid_json_is_logged_without_mqtt_reply(loaded_module, monkeypatch, caplog):
    """No request_id can be recovered from unparseable JSON, so there's no
    topic to key a reply to - by design this only logs, it doesn't publish."""
    m = loaded_module
    caplog.set_level("ERROR")
    published = MagicMock()
    monkeypatch.setattr(m, "client", published)

    m.on_message(MagicMock(), None, _msg(b"{not valid json"))

    assert m.request_queue.qsize() == 0
    published.publish.assert_not_called()
    assert f"error_code={int(m.ErrorCode.INVALID_JSON)}" in caplog.text


def test_on_message_schema_violation_reports_fallback_id(loaded_module, monkeypatch):
    """Missing required fields -> VALIDATION_ERROR, but request_id from the
    raw (unvalidated) dict should still be used so the caller gets an
    error keyed to their request_id, not a silent drop."""
    m = loaded_module
    published = MagicMock()
    monkeypatch.setattr(m, "client", published)

    payload = json.dumps({"request_id": "known-id", "model": "llama3"}).encode("utf-8")
    m.on_message(MagicMock(), None, _msg(payload))

    assert m.request_queue.qsize() == 0
    sent = json.loads(published.publish.call_args.args[1])
    assert sent["request_id"] == "known-id"
    assert sent["error_code"] == int(m.ErrorCode.VALIDATION_ERROR)
    assert "system" in sent["error_message"] or "user" in sent["error_message"]


@pytest.mark.parametrize("raw", [b"[1, 2, 3]", b'"just a string"', b"42", b"null"])
def test_on_message_valid_json_non_dict_root_does_not_crash(loaded_module, raw):
    """JSON that parses fine but isn't an object (list/string/number/null)
    must not blow up on data.get(...) or in pydantic validation."""
    m = loaded_module
    m.on_message(MagicMock(), None, _msg(raw))  # must not raise
    assert m.request_queue.qsize() == 0


def test_on_message_swallows_unexpected_downstream_exception(loaded_module, monkeypatch, caplog):
    """Regression test for the outer safety net: even a bug in code past
    all the specific try/excepts (e.g. request_queue.put itself failing)
    must not escape on_message and kill the paho network thread."""
    m = loaded_module
    monkeypatch.setattr(m.request_queue, "put", MagicMock(side_effect=RuntimeError("queue is broken")))

    payload = json.dumps({
        "request_id": "x", "model": "llama3", "system": "s", "user": "u",
    }).encode("utf-8")

    m.on_message(MagicMock(), None, _msg(payload))  # must not raise

    assert "Unhandled error in on_message" in caplog.text
