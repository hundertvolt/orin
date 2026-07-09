"""
Tests for mqtt_telemetry.py's MQTT client setup (LWT, queue bound) and
its shutdown/error-recovery behavior.

Like mqtt_llm.py's shutdown block, this isn't a function - it's the
tail of the script, running once as part of module exec. Each scenario
needs a fresh module load with different mqtt.Client method mocks (and,
here, different jtop behavior) in place beforehand. See
load_mqtt_telemetry() in conftest.py for how the main loop's blocking
time.sleep() is unblocked without a real signal.
"""
import itertools
import json
import threading
import time
from unittest.mock import MagicMock

import pytest
from conftest import BASELINE_JTOP_FAN, BASELINE_JTOP_STATS, load_mqtt_telemetry


def _connected_then_dropped():
    """is_connected() mock: True on the first call (passes
    wait_for_mqtt_connection()'s startup gate), False on every call
    after - simulating a connection that was fine at startup but has
    dropped by the time the shutdown path checks it."""
    return MagicMock(side_effect=itertools.chain([True], itertools.repeat(False)))


# ------------------------------
# Connection gating before jtop()
# ------------------------------

def test_jtop_never_started_until_connected():
    """Core regression test for wait_for_mqtt_connection(): jtop() must
    not be entered until is_connected() actually returns True - not
    proceed anyway after a bounded grace period (the earlier, looser
    version of this fix)."""
    is_connected = MagicMock(side_effect=[False, False, True] + [True] * 20)

    m = load_mqtt_telemetry(
        ["--broker", "127.0.0.1"],
        client_method_mocks={"is_connected": is_connected},
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )

    assert m.jtop_enter_count == 1
    assert is_connected.call_count >= 3  # polled until it actually turned True


def test_wait_for_mqtt_connection_blocks_until_connected():
    """Core regression test for the check-clear-check race fix: the wait
    must not return early just because connected_event was set - it must
    re-check is_connected() too, and keep blocking for real until the
    connection has actually settled. Called directly (via an already
    fully-loaded-and-shut-down module) rather than through
    load_mqtt_telemetry() itself, since the loader defaults is_connected
    to True and patches threading.Event.wait globally to avoid a real
    stall on every test load - which leaves nothing interesting to
    observe for this race specifically.
    """
    m = load_mqtt_telemetry(
        ["--broker", "127.0.0.1"],
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )
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
        if t.is_alive():
            is_connected_flag.set()
            m.connected_event.set()
            t.join(timeout=2)


def test_wait_for_mqtt_connection_logs_periodically_while_waiting(caplog):
    caplog.set_level("WARNING")
    m = load_mqtt_telemetry(
        ["--broker", "127.0.0.1"],
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )
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
        m.client.is_connected = MagicMock(return_value=True)
        m.connected_event.set()
        t.join(timeout=2)
        assert not t.is_alive()


def test_will_set_registers_offline_lwt_before_connect():
    will_set = MagicMock()
    connect = MagicMock(return_value=None)
    call_order = []
    will_set.side_effect = lambda *a, **k: call_order.append("will_set")
    connect.side_effect = lambda *a, **k: call_order.append("connect")

    m = load_mqtt_telemetry(
        ["--broker", "127.0.0.1", "--topic", "orin/status"],
        client_method_mocks={"will_set": will_set, "connect": connect},
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )

    will_set.assert_called_once_with(
        "orin/status", json.dumps({"heartbeat": 0, "status": "offline"}), qos=1, retain=True
    )
    assert call_order == ["will_set", "connect"]
    assert m.MQTT_TOPIC == "orin/status"


def test_max_queued_messages_bounded_to_five():
    max_queued = MagicMock()
    load_mqtt_telemetry(
        ["--broker", "127.0.0.1"],
        client_method_mocks={"max_queued_messages_set": max_queued},
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )
    max_queued.assert_called_once_with(5)


def test_shutdown_publishes_offline_status_when_connected(caplog):
    caplog.set_level("INFO")
    info = MagicMock()
    publish = MagicMock(return_value=info)
    is_connected = MagicMock(return_value=True)
    loop_stop = MagicMock()
    disconnect = MagicMock()

    load_mqtt_telemetry(
        ["--broker", "127.0.0.1", "--topic", "orin/status", "--loglevel", "DEBUG"],
        client_method_mocks={
            "publish": publish, "is_connected": is_connected,
            "loop_stop": loop_stop, "disconnect": disconnect,
        },
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )

    # publish is called twice: once per the one telemetry cycle, once for
    # the shutdown-time offline status. The offline one is qos=1/retain=True.
    offline_calls = [
        c for c in publish.call_args_list
        if c.args[1] == json.dumps({"heartbeat": 0, "status": "offline"})
    ]
    assert len(offline_calls) == 1
    assert offline_calls[0].kwargs == {"qos": 1, "retain": True}
    info.wait_for_publish.assert_called_once_with(timeout=2.0)
    loop_stop.assert_called_once()
    disconnect.assert_called_once()
    assert "Shutdown complete." in caplog.text


def test_shutdown_skips_offline_publish_when_not_connected(caplog):
    caplog.set_level("INFO")
    publish = MagicMock()
    is_connected = _connected_then_dropped()
    loop_stop = MagicMock()
    disconnect = MagicMock()

    load_mqtt_telemetry(
        ["--broker", "127.0.0.1"],
        client_method_mocks={
            "publish": publish, "is_connected": is_connected,
            "loop_stop": loop_stop, "disconnect": disconnect,
        },
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )

    publish.assert_not_called()
    loop_stop.assert_called_once()
    disconnect.assert_called_once()
    assert "Shutdown complete." in caplog.text


def test_shutdown_survives_loop_stop_exception(caplog):
    """The shutdown block wraps the offline-publish + loop_stop()/
    disconnect() sequence in one try/except - a failure partway through
    must not prevent 'Shutdown complete.' from being logged."""
    caplog.set_level("INFO")
    is_connected = _connected_then_dropped()
    loop_stop = MagicMock(side_effect=RuntimeError("thread join failed"))
    disconnect = MagicMock()

    load_mqtt_telemetry(
        ["--broker", "127.0.0.1"],
        client_method_mocks={
            "is_connected": is_connected, "loop_stop": loop_stop, "disconnect": disconnect,
        },
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )  # must not raise out of exec_module

    loop_stop.assert_called_once()
    disconnect.assert_not_called()  # same try block: the exception short-circuits it
    assert "Error during MQTT shutdown" in caplog.text
    assert "Shutdown complete." in caplog.text


def test_shutdown_survives_disconnect_exception(caplog):
    caplog.set_level("INFO")
    is_connected = _connected_then_dropped()
    loop_stop = MagicMock()
    disconnect = MagicMock(side_effect=OSError("socket already gone"))

    load_mqtt_telemetry(
        ["--broker", "127.0.0.1"],
        client_method_mocks={
            "is_connected": is_connected, "loop_stop": loop_stop, "disconnect": disconnect,
        },
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )  # must not raise out of exec_module

    assert "Error during MQTT shutdown" in caplog.text
    assert "Shutdown complete." in caplog.text


# ------------------------------
# "let systemd restart" behavior
# ------------------------------

def test_jtop_failure_reraises_for_systemd_restart(caplog):
    """Regression test for the fix: a fatal main-loop error (jtop()
    failing to open) must propagate out instead of exiting cleanly, so
    a systemd Restart=on-failure policy actually restarts the service."""
    caplog.set_level("INFO")
    error = RuntimeError("jetson_stats service is not running (simulated)")

    with pytest.raises(RuntimeError, match="jetson_stats service is not running"):
        load_mqtt_telemetry(
            ["--broker", "127.0.0.1"],
            jtop_enter_error=error,
        )

    assert "Error in main loop" in caplog.text


def test_jtop_failure_still_publishes_offline_and_cleans_up(caplog):
    """The fatal-error path must still run the normal shutdown sequence
    (offline status, loop_stop, disconnect) before propagating - the
    MQTT client was already connected by the time jtop() failed."""
    caplog.set_level("INFO")
    info = MagicMock()
    publish = MagicMock(return_value=info)
    is_connected = MagicMock(return_value=True)
    loop_stop = MagicMock()
    disconnect = MagicMock()

    with pytest.raises(RuntimeError):
        load_mqtt_telemetry(
            ["--broker", "127.0.0.1", "--topic", "orin/status"],
            client_method_mocks={
                "publish": publish, "is_connected": is_connected,
                "loop_stop": loop_stop, "disconnect": disconnect,
            },
            jtop_enter_error=RuntimeError("simulated jtop failure"),
        )

    publish.assert_called_once_with(
        "orin/status", json.dumps({"heartbeat": 0, "status": "offline"}), qos=1, retain=True
    )
    loop_stop.assert_called_once()
    disconnect.assert_called_once()
    assert "Shutdown complete." in caplog.text


def test_jtop_is_actually_invoked_during_load():
    """Sanity check on the test harness itself: the happy-path loader
    fixture really did call jtop().__enter__() once."""
    m = load_mqtt_telemetry(
        ["--broker", "127.0.0.1"],
        jtop_stats=BASELINE_JTOP_STATS, jtop_fan=BASELINE_JTOP_FAN,
    )
    assert m.jtop_enter_count == 1
