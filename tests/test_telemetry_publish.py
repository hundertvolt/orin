"""
Unit tests for publish_telemetry()'s payload generation and publish
behavior: connected/not-connected gating, missing/partial jtop stats,
the -256 "sensor absent" sentinel, and that nothing inside can escape
and crash the main loop.
"""
import json
from unittest.mock import MagicMock

from conftest import BASELINE_JTOP_FAN, BASELINE_JTOP_STATS, FakeJetson, RaisingFakeJetson


def _publish_and_get_payload(module, jetson, monkeypatch):
    client = MagicMock()
    client.publish.return_value = MagicMock(rc=0)
    client.is_connected.return_value = True
    monkeypatch.setattr(module, "client", client)

    module.publish_telemetry(jetson)

    client.publish.assert_called_once()
    args, kwargs = client.publish.call_args
    assert args[0] == module.MQTT_TOPIC
    return json.loads(args[1]), kwargs


def test_happy_path_full_stats(loaded_telemetry_module, monkeypatch):
    m = loaded_telemetry_module
    payload, kwargs = _publish_and_get_payload(
        m, FakeJetson(BASELINE_JTOP_STATS, BASELINE_JTOP_FAN), monkeypatch
    )

    assert payload["status"] == "online"
    assert payload["cpu_avg"] == (5.0 + 8.0 + 3.0 + 6.0) / 4
    assert payload["cpu_max"] == 8.0
    assert payload["ram_used_ratio"] == 0.35
    assert payload["swap_used_ratio"] == 0.0
    assert payload["gpu_load"] == 12.5
    assert payload["fan_pwm"] == 60
    assert payload["fan_rpm"] == 1500
    assert payload["temp_cpu"] == 45.0
    assert payload["temp_gpu"] == 44.0
    assert payload["power_total"] == 4.2  # 4200 * 1e-3
    assert payload["uptime_s"] == 3725
    assert isinstance(payload["heartbeat"], int)
    assert kwargs.get("qos") == 1
    assert kwargs.get("retain") is False


def test_temp_max_excludes_sentinel_but_ignores_unfiltered_zone_temps(loaded_telemetry_module, monkeypatch):
    """The -256 'sensor absent' sentinel is filtered out of temp_max
    (which scans all "Temp *" keys), so a bogus -256 reading from an
    absent zone (e.g. "Temp tj" in the baseline stats) doesn't corrupt
    the max. The real max among present sensors (45.0 from CPU) wins."""
    m = loaded_telemetry_module
    payload, _ = _publish_and_get_payload(
        m, FakeJetson(BASELINE_JTOP_STATS, BASELINE_JTOP_FAN), monkeypatch
    )
    assert payload["temp_max"] == 45.0


def test_temp_cpu_not_filtered_for_sentinel(loaded_telemetry_module, monkeypatch):
    """Documents existing behavior distinct from temp_max: temp_cpu and
    temp_gpu are read directly via stats.get(...) without the -256
    filter that temp_max applies, so an absent-sensor reading of -256
    passes straight through into the payload as -256, not None."""
    m = loaded_telemetry_module
    stats = dict(BASELINE_JTOP_STATS, **{"Temp cpu": -256})
    payload, _ = _publish_and_get_payload(m, FakeJetson(stats, BASELINE_JTOP_FAN), monkeypatch)
    assert payload["temp_cpu"] == -256


def test_no_cpu_cores_present(loaded_telemetry_module, monkeypatch):
    m = loaded_telemetry_module
    stats = {k: v for k, v in BASELINE_JTOP_STATS.items() if not k.startswith("CPU")}
    payload, _ = _publish_and_get_payload(m, FakeJetson(stats, BASELINE_JTOP_FAN), monkeypatch)
    assert payload["cpu_avg"] is None
    assert payload["cpu_max"] is None


def test_no_temp_readings_present(loaded_telemetry_module, monkeypatch):
    m = loaded_telemetry_module
    stats = {k: v for k, v in BASELINE_JTOP_STATS.items() if not k.startswith("Temp")}
    payload, _ = _publish_and_get_payload(m, FakeJetson(stats, BASELINE_JTOP_FAN), monkeypatch)
    assert payload["temp_max"] is None
    assert payload["temp_cpu"] is None
    assert payload["temp_gpu"] is None


def test_missing_uptime(loaded_telemetry_module, monkeypatch):
    m = loaded_telemetry_module
    stats = {k: v for k, v in BASELINE_JTOP_STATS.items() if k != "uptime"}
    payload, _ = _publish_and_get_payload(m, FakeJetson(stats, BASELINE_JTOP_FAN), monkeypatch)
    assert payload["uptime_s"] is None


def test_missing_power(loaded_telemetry_module, monkeypatch):
    m = loaded_telemetry_module
    stats = {k: v for k, v in BASELINE_JTOP_STATS.items() if k != "Power TOT"}
    payload, _ = _publish_and_get_payload(m, FakeJetson(stats, BASELINE_JTOP_FAN), monkeypatch)
    assert payload["power_total"] is None


def test_missing_fan_data(loaded_telemetry_module, monkeypatch):
    m = loaded_telemetry_module
    payload, _ = _publish_and_get_payload(m, FakeJetson(BASELINE_JTOP_STATS, {}), monkeypatch)
    assert payload["fan_pwm"] is None
    assert payload["fan_rpm"] is None


def test_missing_ram_gpu_swap(loaded_telemetry_module, monkeypatch):
    m = loaded_telemetry_module
    stats = {k: v for k, v in BASELINE_JTOP_STATS.items() if k not in ("RAM", "SWAP", "GPU")}
    payload, _ = _publish_and_get_payload(m, FakeJetson(stats, BASELINE_JTOP_FAN), monkeypatch)
    assert payload["ram_used_ratio"] is None
    assert payload["swap_used_ratio"] is None
    assert payload["gpu_load"] is None


def test_skips_publish_when_not_connected(loaded_telemetry_module, monkeypatch, caplog):
    m = loaded_telemetry_module
    caplog.set_level("DEBUG")
    client = MagicMock()
    client.is_connected.return_value = False
    monkeypatch.setattr(m, "client", client)

    m.publish_telemetry(FakeJetson(BASELINE_JTOP_STATS, BASELINE_JTOP_FAN))

    client.publish.assert_not_called()
    assert "Not publishing; client is not connected" in caplog.text


def test_publish_error_code_is_logged_without_crashing(loaded_telemetry_module, monkeypatch, caplog):
    m = loaded_telemetry_module
    client = MagicMock()
    client.is_connected.return_value = True
    client.publish.return_value = MagicMock(rc=1)  # MQTT_ERR_NO_CONN-ish
    monkeypatch.setattr(m, "client", client)

    m.publish_telemetry(FakeJetson(BASELINE_JTOP_STATS, BASELINE_JTOP_FAN))  # must not raise

    assert "Publish returned error code" in caplog.text


def test_stats_access_exception_is_contained(loaded_telemetry_module, monkeypatch, caplog):
    """If the jtop backend itself misbehaves (e.g. stats access raises),
    the whole main loop must not go down with it."""
    m = loaded_telemetry_module
    client = MagicMock()
    client.is_connected.return_value = True
    monkeypatch.setattr(m, "client", client)

    m.publish_telemetry(RaisingFakeJetson(RuntimeError("jtop backend crashed")))  # must not raise

    client.publish.assert_not_called()
    assert "Telemetry publish error" in caplog.text


def test_publish_call_itself_raising_is_contained(loaded_telemetry_module, monkeypatch, caplog):
    m = loaded_telemetry_module
    client = MagicMock()
    client.is_connected.return_value = True
    client.publish.side_effect = OSError("no route to host")
    monkeypatch.setattr(m, "client", client)

    m.publish_telemetry(FakeJetson(BASELINE_JTOP_STATS, BASELINE_JTOP_FAN))  # must not raise

    assert "Telemetry publish error" in caplog.text
