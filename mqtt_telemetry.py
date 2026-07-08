#!/usr/bin/env python3

import argparse
import json
import logging
import signal
import time

import paho.mqtt.client as mqtt
from jtop import jtop
from pathlib import Path
from typing import Any, Dict, Optional

# ------------------------------
# Command-line arguments
# ------------------------------

logChoices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
parser = argparse.ArgumentParser(description="Orin Nano telemetry to MQTT")
parser.add_argument("--broker", required=True, help="MQTT broker IP or hostname")
parser.add_argument("--port", type=int, default=1883, help="MQTT broker port (default 1883)")
parser.add_argument("--username", help="MQTT username")
parser.add_argument("--credpath", help="LoadCredential path for MQTT password")
parser.add_argument("--topic", default="orin/status", help="MQTT topic to publish telemetry")
parser.add_argument("--interval", type=int, default=10, help="Publish interval in seconds")
parser.add_argument("--loglevel", default="INFO", choices=logChoices, help="Logging level")

args = parser.parse_args()

MQTT_BROKER: str = args.broker
MQTT_PORT: int = args.port
MQTT_TOPIC: str = args.topic
PUBLISH_INTERVAL: int = args.interval
USERNAME: Optional[str] = args.username
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

def on_connect(cli: mqtt.Client, userdata: Any, flags: Dict[str, int], reason_code: int, properties: Any) -> None:
    # paho re-raises exceptions escaping this callback, which kills its network
    # loop thread for good, so every path out of here must be caught locally.
    try:
        if reason_code == 0:
            log.info("Connected to MQTT broker")
        else:
            log.error(f"MQTT connection failed: {reason_code}")
    except Exception as e:
        log.error(f"Unhandled error in on_connect: {e}")

def on_disconnect(cli: mqtt.Client, userdata: Any, flags: Dict[str, int], reason_code: int, properties: Any) -> None:
    try:
        if reason_code != 0:
            log.warning(f"Unexpected MQTT disconnection: {reason_code}")
        else:
            log.info("MQTT disconnected cleanly")
    except Exception as e:
        log.error(f"Unhandled error in on_disconnect: {e}")

client = mqtt.Client(client_id="orin_nano", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
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

client.loop_start()  # background loop handles reconnects

# ------------------------------
# Telemetry publishing
# ------------------------------

def publish_telemetry(jetson: jtop) -> None:
    try:
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

        payload_str = json.dumps(payload)
        log.debug(f"Generated payload: {payload_str}")
        if client.is_connected():
            log.debug(f"Publishing to topic {MQTT_TOPIC}")
            info = client.publish(MQTT_TOPIC, payload_str, qos=1, retain=False)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                log.warning(f"Publish returned error code: {info.rc}")
        else:
            log.debug(f"Not publishing; client is not connected!")
    except Exception as e:
        log.error(f"Telemetry publish error: {e}")


def handle_exit(signum: int, frame: Optional[Any]) -> None:
    log.info(f"Received signal {signum}, shutting down...")
    raise SystemExit

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

# ------------------------------
# Main loop using jtop
# ------------------------------

try:
    with jtop() as jetsonTop:
        log.info(f"Telemetry service started using topic: {MQTT_TOPIC}.")
        while True:
            publish_telemetry(jetsonTop)
            time.sleep(PUBLISH_INTERVAL)

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
