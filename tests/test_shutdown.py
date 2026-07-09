"""
Tests for MQTT client setup (LWT, queue bound) and the shutdown block.

mqtt_llm.py's shutdown sequence isn't a function - it's the tail of the
script, running once as part of module exec (see load_mqtt_llm() in
conftest.py for how shutdown_event.wait() is triggered without
blocking the import). Each scenario therefore needs a fresh module
load with different mqtt.Client method mocks in place beforehand.
"""
import json
import threading
import time
from unittest.mock import MagicMock

from conftest import load_mqtt_llm


def test_will_set_registers_offline_lwt_before_connect():
    will_set = MagicMock()
    connect = MagicMock(return_value=None)
    call_order = []
    will_set.side_effect = lambda *a, **k: call_order.append("will_set")
    connect.side_effect = lambda *a, **k: call_order.append("connect")

    m = load_mqtt_llm(
        ["--broker", "127.0.0.1", "--topic", "orin/ollama"],
        client_method_mocks={"will_set": will_set, "connect": connect},
    )

    will_set.assert_called_once_with(
        "orin/ollama/status", json.dumps({"status": "offline"}), qos=1, retain=True
    )
    assert call_order == ["will_set", "connect"]
    assert m.MQTT_STATUS_TOPIC == "orin/ollama/status"


def test_max_queued_messages_bounded_to_five():
    max_queued = MagicMock()

    load_mqtt_llm(["--broker", "127.0.0.1"], client_method_mocks={"max_queued_messages_set": max_queued})

    max_queued.assert_called_once_with(5)


def test_shutdown_publishes_offline_status_when_connected(caplog):
    caplog.set_level("INFO")
    info = MagicMock()
    publish = MagicMock(return_value=info)
    is_connected = MagicMock(return_value=True)
    loop_stop = MagicMock()
    disconnect = MagicMock()

    load_mqtt_llm(
        ["--broker", "127.0.0.1", "--topic", "orin/ollama", "--loglevel", "DEBUG"],
        client_method_mocks={
            "publish": publish,
            "is_connected": is_connected,
            "loop_stop": loop_stop,
            "disconnect": disconnect,
        },
    )

    publish.assert_called_once_with(
        "orin/ollama/status", json.dumps({"status": "offline"}), qos=1, retain=True
    )
    info.wait_for_publish.assert_called_once_with(timeout=2.0)
    loop_stop.assert_called_once()
    disconnect.assert_called_once()
    assert "Shutdown complete." in caplog.text


def test_shutdown_skips_offline_publish_when_not_connected(caplog):
    caplog.set_level("INFO")
    publish = MagicMock()
    is_connected = MagicMock(return_value=False)
    loop_stop = MagicMock()
    disconnect = MagicMock()

    load_mqtt_llm(
        ["--broker", "127.0.0.1"],
        client_method_mocks={
            "publish": publish,
            "is_connected": is_connected,
            "loop_stop": loop_stop,
            "disconnect": disconnect,
        },
    )

    publish.assert_not_called()
    loop_stop.assert_called_once()
    disconnect.assert_called_once()
    assert "Shutdown complete." in caplog.text


def test_shutdown_survives_loop_stop_exception(caplog):
    """The original bug: an unguarded loop_stop()/disconnect() failure
    would abort the shutdown sequence before 'Shutdown complete.' logs,
    and before disconnect() ever runs."""
    caplog.set_level("INFO")
    is_connected = MagicMock(return_value=False)
    loop_stop = MagicMock(side_effect=RuntimeError("thread join failed"))
    disconnect = MagicMock()

    load_mqtt_llm(
        ["--broker", "127.0.0.1"],
        client_method_mocks={
            "is_connected": is_connected,
            "loop_stop": loop_stop,
            "disconnect": disconnect,
        },
    )  # must not raise out of exec_module

    loop_stop.assert_called_once()
    disconnect.assert_not_called()  # same try block: the exception short-circuits it
    assert "Error while stopping MQTT client" in caplog.text
    assert "Shutdown complete." in caplog.text


def test_shutdown_survives_disconnect_exception(caplog):
    caplog.set_level("INFO")
    is_connected = MagicMock(return_value=False)
    loop_stop = MagicMock()
    disconnect = MagicMock(side_effect=OSError("socket already gone"))

    load_mqtt_llm(
        ["--broker", "127.0.0.1"],
        client_method_mocks={
            "is_connected": is_connected,
            "loop_stop": loop_stop,
            "disconnect": disconnect,
        },
    )  # must not raise out of exec_module

    assert "Error while stopping MQTT client" in caplog.text
    assert "Shutdown complete." in caplog.text


def test_shutdown_survives_wait_for_publish_timeout(caplog):
    """wait_for_publish can itself raise (e.g. ValueError on a bad timeout,
    or if the message was rejected outright) - must not abort shutdown."""
    caplog.set_level("INFO")
    info = MagicMock()
    info.wait_for_publish.side_effect = RuntimeError("publish never acked")
    publish = MagicMock(return_value=info)
    is_connected = MagicMock(return_value=True)
    loop_stop = MagicMock()
    disconnect = MagicMock()

    load_mqtt_llm(
        ["--broker", "127.0.0.1"],
        client_method_mocks={
            "publish": publish,
            "is_connected": is_connected,
            "loop_stop": loop_stop,
            "disconnect": disconnect,
        },
    )  # must not raise

    loop_stop.assert_not_called()  # short-circuited by the exception, same as loop_stop failing
    assert "Error while stopping MQTT client" in caplog.text
    assert "Shutdown complete." in caplog.text


# ------------------------------
# wait_for_mqtt_connection()
# ------------------------------
#
# Called directly on an already-loaded module rather than through
# load_mqtt_llm() itself, since the loader unblocks both shutdown_event
# and connected_event immediately to avoid a real 5s stall on every test
# load (see load_mqtt_llm()'s docstring in conftest.py) - which makes
# the wait resolve before there's anything interesting to observe.
# Calling the function directly, with a fresh module and full control
# over client.is_connected()/the two events, exercises the real race
# fix in isolation instead.

def test_wait_for_mqtt_connection_returns_immediately_if_already_connected(loaded_module):
    m = loaded_module
    m.client.is_connected = MagicMock(return_value=True)

    start = time.monotonic()
    m.wait_for_mqtt_connection()

    assert time.monotonic() - start < 0.5


def test_wait_for_mqtt_connection_returns_when_shutdown_requested(loaded_module):
    m = loaded_module
    m.client.is_connected = MagicMock(return_value=False)
    m.shutdown_event.set()
    try:
        start = time.monotonic()
        m.wait_for_mqtt_connection()
        assert time.monotonic() - start < 0.5
    finally:
        m.shutdown_event.clear()


def test_wait_for_mqtt_connection_blocks_until_connected(loaded_module):
    """Core regression test: the wait must not return early just because
    connected_event was set - it must re-check is_connected() too, and
    keep blocking for real until the connection has actually settled."""
    m = loaded_module
    is_connected_flag = threading.Event()
    m.client.is_connected = MagicMock(side_effect=is_connected_flag.is_set)

    result_holder = {}

    def run():
        m.wait_for_mqtt_connection()
        result_holder["done"] = True

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        time.sleep(0.2)
        assert "done" not in result_holder, "returned before the connection ever settled"

        is_connected_flag.set()
        m.connected_event.set()
        t.join(timeout=2)

        assert not t.is_alive()
        assert result_holder.get("done") is True
    finally:
        m.shutdown_event.set()
        m.connected_event.set()
        t.join(timeout=2)
        m.shutdown_event.clear()


def test_wait_for_mqtt_connection_logs_periodically_while_waiting(loaded_module, caplog):
    caplog.set_level("WARNING")
    m = loaded_module
    m.CONNECT_RETRY_LOG_INTERVAL = 0.05
    m.client.is_connected = MagicMock(return_value=False)

    t = threading.Thread(target=m.wait_for_mqtt_connection, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "Still waiting for MQTT connection" not in caplog.text:
            time.sleep(0.02)
        assert "Still waiting for MQTT connection" in caplog.text
    finally:
        m.shutdown_event.set()
        m.connected_event.set()
        t.join(timeout=2)
        assert not t.is_alive()
        m.shutdown_event.clear()
