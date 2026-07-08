"""
Unit tests for mqtt_telemetry.py's on_connect / on_disconnect.

Same paho-mqtt hazard as mqtt_llm.py: paho re-raises exceptions
escaping a callback by default (suppress_exceptions=False), and none of
its internal _loop()/loop_forever()/_thread_main() catch it - an
uncaught exception here kills the network loop thread forever while the
process keeps running. Since this script never re-subscribes to
anything, the practical effect is telemetry silently stopping forever:
client.is_connected() stays False on every subsequent tick, with only a
DEBUG-level "not connected" line as a trace. The core regression
coverage here is: no matter what breaks inside these callbacks, the
call itself must always return normally.
"""
from unittest.mock import MagicMock, patch


def test_on_connect_success_logs_info(loaded_telemetry_module, caplog):
    m = loaded_telemetry_module
    caplog.set_level("INFO")
    m.on_connect(MagicMock(), None, {}, 0, None)
    assert "Connected to MQTT broker" in caplog.text


def test_on_connect_failure_logs_error(loaded_telemetry_module, caplog):
    m = loaded_telemetry_module
    m.on_connect(MagicMock(), None, {}, 5, None)
    assert "MQTT connection failed" in caplog.text


def test_on_connect_swallows_internal_exception(loaded_telemetry_module, caplog):
    """Regression test for the paho-thread-killing bug: a failure inside
    on_connect's own logging call must not escape."""
    m = loaded_telemetry_module
    with patch.object(m.log, "info", MagicMock(side_effect=RuntimeError("boom"))):
        m.on_connect(MagicMock(), None, {}, 0, None)  # must not raise
    assert "Unhandled error in on_connect" in caplog.text


def test_on_disconnect_clean(loaded_telemetry_module, caplog):
    m = loaded_telemetry_module
    caplog.set_level("INFO")
    m.on_disconnect(MagicMock(), None, {}, 0, None)
    assert "MQTT disconnected cleanly" in caplog.text


def test_on_disconnect_unexpected(loaded_telemetry_module, caplog):
    m = loaded_telemetry_module
    m.on_disconnect(MagicMock(), None, {}, 7, None)
    assert "Unexpected MQTT disconnection" in caplog.text


def test_on_disconnect_swallows_internal_exception(loaded_telemetry_module, caplog):
    m = loaded_telemetry_module
    with patch.object(m.log, "warning", MagicMock(side_effect=RuntimeError("boom"))):
        m.on_disconnect(MagicMock(), None, {}, 7, None)  # must not raise
    assert "Unhandled error in on_disconnect" in caplog.text
