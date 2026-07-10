#!/usr/bin/env python3

import argparse
import json
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
from jtop import jtop

# ------------------------------
# Command-line arguments
# ------------------------------

def _port_type(value: str) -> int:
    ivalue = int(value)
    if not (1 <= ivalue <= 65535):
        raise argparse.ArgumentTypeError(f"must be between 1 and 65535, got {ivalue}")
    return ivalue

def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {ivalue}")  # else time.sleep() fails later, not now
    return ivalue

logChoices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
parser = argparse.ArgumentParser(description="Orin Nano telemetry to MQTT")
parser.add_argument("--broker", required=True, help="MQTT broker IP or hostname")
parser.add_argument("--port", type=_port_type, default=1883, help="MQTT broker port (default 1883)")
parser.add_argument("--username", help="MQTT username")
parser.add_argument("--credpath", help="LoadCredential path for MQTT password")
parser.add_argument("--topic", default="orin/status", help="MQTT topic to publish telemetry")
parser.add_argument("--interval", type=_positive_int, default=10, help="Publish interval in seconds")
parser.add_argument("--loglevel", default="INFO", choices=logChoices, help="Logging level")

args = parser.parse_args()

MQTT_BROKER: str = args.broker
MQTT_PORT: int = args.port
MQTT_TOPIC: str = args.topic
PUBLISH_INTERVAL: int = args.interval
USERNAME: str | None = args.username
CRED_PATH = Path(args.credpath) if args.credpath else None

# ------------------------------
# Logging setup
# ------------------------------

logging.basicConfig(
    level=getattr(logging, args.loglevel.upper(), logging.INFO),
    format="[%(levelname)s] [%(name)s] %(message)s"
)

log = logging.getLogger("orin_mqtt")

# ------------------------------
# MQTT setup
# ------------------------------

CONNECT_RETRY_LOG_INTERVAL = 5.0  # how often to log while waiting for a connection

connected_event = threading.Event()

def on_connect(
    cli: mqtt.Client,
    userdata: Any,
    flags: mqtt.ConnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    # paho re-raises exceptions escaping this callback, which kills its network
    # loop thread for good, so every path out of here must be caught locally.
    try:
        if reason_code == 0:
            log.info("Connected to MQTT broker")
        else:
            log.error(f"MQTT connection failed: {reason_code}")
    except Exception as e:
        log.error(f"Unhandled error in on_connect: {e}")
    finally:
        connected_event.set()

def on_disconnect(
    cli: mqtt.Client,
    userdata: Any,
    flags: mqtt.DisconnectFlags,
    reason_code: mqtt.ReasonCode,
    properties: mqtt.Properties | None,
) -> None:
    try:
        if reason_code != 0:
            log.warning(f"Unexpected MQTT disconnection: {reason_code}")
        else:
            log.info("MQTT disconnected cleanly")
    except Exception as e:
        log.error(f"Unhandled error in on_disconnect: {e}")

# Block until genuinely connected, not just connect() called, before jtop() ever starts.
# See README.md#telemetry-connect-race for the race this closes.
def wait_for_mqtt_connection() -> None:
    while not client.is_connected():
        connected_event.clear()
        if client.is_connected():
            continue  # connected in the instant between the check above and clear()
        if not connected_event.wait(timeout=CONNECT_RETRY_LOG_INTERVAL):
            log.warning(
                f"Still waiting for MQTT connection after {CONNECT_RETRY_LOG_INTERVAL}s; "
                "retrying in the background..."
            )

client = mqtt.Client(client_id="orin_telemetry", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.max_queued_messages_set(5)
client.will_set(MQTT_TOPIC, json.dumps({"heartbeat": 0, "status": "offline"}), qos=1, retain=True)
client.reconnect_delay_set(min_delay=1, max_delay=30)
client.on_connect = on_connect
client.on_disconnect = on_disconnect

log.info(f"Trying to connect to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
if USERNAME:
    PASSWORD = None
    if CRED_PATH:
        try:
            PASSWORD = CRED_PATH.read_text().strip()
        except Exception as e:
            log.error(f"Failed to read MQTT credential file {CRED_PATH}: {e}")
    if PASSWORD is None:
        log.warning("MQTT password not provided; attempting connection without password.")
    client.username_pw_set(USERNAME, PASSWORD)
    log.debug("Using authentication for login.")

try:
    client.connect(MQTT_BROKER, MQTT_PORT)
except Exception as e:
    log.error(f"Initial MQTT connect failed: {e}")
    raise  # let systemd restart

try:
    client.loop_start()  # background loop handles reconnects
except Exception as e:
    log.error(f"Failed to start MQTT network loop: {e}")
    raise  # let systemd restart

def handle_exit(signum: int, frame: Any | None) -> None:
    log.info(f"Received signal {signum}, shutting down...")
    raise SystemExit

# Registered before wait_for_mqtt_connection() so a signal can interrupt that wait too.
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

# ------------------------------
# Telemetry publishing
# ------------------------------

_OFFLINE_FIELDS = {
    "uptime_s": None, "cpu_avg": None, "cpu_max": None,
    "ram_used_ratio": None, "swap_used_ratio": None, "gpu_load": None,
    "fan_pwm": None, "fan_rpm": None,
    "temp_max": None, "temp_cpu": None, "temp_gpu": None,
    "power_total": None,
}


def _publish_payload(payload: dict[str, Any]) -> None:
    payload_str = json.dumps(payload)
    log.debug(f"Generated payload: {payload_str}")
    try:
        if client.is_connected():
            log.debug(f"Publishing to topic {MQTT_TOPIC}")
            info = client.publish(MQTT_TOPIC, payload_str, qos=1, retain=False)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                log.warning(f"Publish returned error code: {info.rc}")
        else:
            log.debug("Not publishing; client is not connected!")
    except Exception as e:
        log.error(f"Telemetry publish error: {e}")


# Build a telemetry payload from jtop and publish it over MQTT.
# ok(spin=True)/.stats/.fan are left unguarded on purpose - exceptions here
# propagate to the main loop's jtop retry wrapper. See README.md#jtop-lifecycle.
def publish_telemetry(jetson: jtop) -> None:
    jetson.ok(spin=True)

    stats = jetson.stats

    # --- CPU handling ---
    cpu_cores = {k: v for k, v in stats.items() if k.startswith("CPU")}

    # --- Temperature handling (filter invalid -256) ---
    temps = {k: v for k, v in stats.items() if k.startswith("Temp ") and v != -256}

    # Fan handling
    fan = jetson.fan.get("pwmfan", {})

    payload = {
        "heartbeat": int(time.time()),
        "uptime_s": int(uptime.total_seconds()) if (uptime := stats.get("uptime")) is not None else None,
        "cpu_avg": sum(cpu_cores.values()) / len(cpu_cores) if cpu_cores else None,
        "cpu_max": max(cpu_cores.values()) if cpu_cores else None,
        "ram_used_ratio": stats.get("RAM"),
        "swap_used_ratio": stats.get("SWAP"),
        "gpu_load": stats.get("GPU"),
        "fan_pwm": fan.get("speed", [None])[0],
        "fan_rpm": fan.get("rpm", [None])[0],
        "temp_max": max(temps.values()) if temps else None,
        "temp_cpu": stats.get("Temp cpu"),
        "temp_gpu": stats.get("Temp gpu"),
        "power_total": power * 1e-3 if (power := stats.get("Power TOT")) is not None else None,
        "status": "online",
    }

    _publish_payload(payload)


# Reports a jtop outage on the telemetry topic (real heartbeat, null sensors) -
# distinct from the heartbeat:0 LWT/shutdown message. See README.md#jtop-lifecycle.
def publish_offline_telemetry() -> None:
    _publish_payload({"heartbeat": int(time.time()), "status": "offline", **_OFFLINE_FIELDS})


# ------------------------------
# Main loop using jtop
# ------------------------------

JTOP_RETRY_DELAY_MIN = 5.0   # initial delay before retrying after a jtop failure
JTOP_RETRY_DELAY_MAX = 60.0  # cap for the backoff below, reached after repeated failures


# Runs jetson.close() in a daemon thread, since it can block on a locked dpkg.
# See README.md#jtop-lifecycle. Don't call jetson.close() synchronously here.
def _close_jtop_in_background(jetson: jtop) -> None:
    def run() -> None:
        try:
            jetson.close()
        except Exception as e:
            log.warning(f"Error closing jtop: {e}")

    threading.Thread(target=run, name="jtop-close", daemon=True).start()


try:
    wait_for_mqtt_connection()
    log.info(f"Telemetry service started using topic: {MQTT_TOPIC}.")

    retry_delay = JTOP_RETRY_DELAY_MIN
    while True:
        # jtop() failing to open or breaking mid-run are handled identically: discard and retry.
        # SystemExit isn't an Exception subclass, so it passes through untouched. See README.md#jtop-lifecycle.
        try:
            with jtop() as jetsonTop:
                log.info("Connected to jtop.")
                retry_delay = JTOP_RETRY_DELAY_MIN  # reachable again - reset the backoff
                try:
                    while True:
                        publish_telemetry(jetsonTop)
                        time.sleep(PUBLISH_INTERVAL)
                finally:
                    _close_jtop_in_background(jetsonTop)
        except Exception as e:
            log.error(f"jtop error, retrying in {retry_delay:.0f}s: {e}")
            publish_offline_telemetry()
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, JTOP_RETRY_DELAY_MAX)  # back off - a fresh jtop() isn't free

except SystemExit:
    log.debug("SystemExit received, stopping main loop.")

except Exception as e:
    log.error(f"Error in main loop: {e}")
    raise  # let systemd restart

finally:
    log.info("Stopping MQTT loop and disconnecting...")
    try:
        if client.is_connected():
            info = client.publish(MQTT_TOPIC, json.dumps({"heartbeat": 0, "status": "offline"}), qos=1, retain=True)
            info.wait_for_publish(timeout=2.0)  # ensure the message is sent
            log.debug("Offline payload published.")
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        log.error(f"Error during MQTT shutdown: {e}")
    log.info("Shutdown complete.")
