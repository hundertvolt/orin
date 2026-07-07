"""
Tests for MQTT client setup (LWT, queue bound) and the shutdown block.

mqtt_llm.py's shutdown sequence isn't a function - it's the tail of the
script, running once as part of module exec (see load_mqtt_llm() in
conftest.py for how shutdown_event.wait() is triggered without
blocking the import). Each scenario therefore needs a fresh module
load with different mqtt.Client method mocks in place beforehand.
"""
import json
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

    m = load_mqtt_llm(
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
