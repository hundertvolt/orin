#!/usr/bin/env python3

import argparse
import json
import logging
import signal
import time
import requests

import paho.mqtt.client as mqtt
from jtop import jtop
from pathlib import Path
from typing import Any, Dict, Optional

# ------------------------------
# Command-line arguments
# ------------------------------

logChoices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
parser = argparse.ArgumentParser(description="MQTT interface to Ollama")
parser.add_argument("--broker", required=True, help="MQTT broker IP or hostname")
parser.add_argument("--port", type=int, default=1883, help="MQTT broker port (default 1883)")
parser.add_argument("--username", help="MQTT username")
parser.add_argument("--credpath", help="LoadCredential path for MQTT password")
parser.add_argument("--topic", default="orin/ollama", help="MQTT topic to publish and receive")
parser.add_argument("--ollama-host", default="localhost", help="Ollama host address (default: localhost)")
parser.add_argument("--ollama-port", type=int, default="11434", help="Ollama host address (default: 11434)")

args = parser.parse_args()

MQTT_BROKER: str = args.broker
MQTT_PORT: int = args.port
MQTT_TOPIC: str = args.topic
USERNAME: Optional[str] = args.username
CRED_PATH = Path(args.credpath) if args.credpath else None
OLLAMA_URL: str = f"http://{args.ollama_host}:{args.ollama_port}/api/generate"

# ------------------------------
# Logging setup
# ------------------------------

logging.basicConfig(
    level=getattr(logging, args.loglevel.upper(), logging.INFO),
    format="[%(levelname)s] [%(name)s] %(message)s"
)

log = logging.getLogger("ollama_mqtt")

# ------------------------------
# MQTT setup
# ------------------------------

def on_connect(cli: mqtt.Client, userdata: Any, flags: Dict[str, int], reason_code: int, properties: Any) -> None:
    if reason_code == 0:
        sub_topic = f"{MQTT_TOPIC}/request"
        log.info(f"Connected to MQTT broker, listening to '{sub_topic}'.")
        cli.subscribe(sub_topic, qos=1)
    else:
        log.error(f"MQTT connection failed: {reason_code}")

def on_disconnect(cli: mqtt.Client, userdata: Any, flags: Dict[str, int], reason_code: int, properties: Any) -> None:
    if reason_code != 0:
        log.warning(f"Unexpected MQTT disconnection: {reason_code}")
    else:
        log.info("MQTT disconnected cleanly")

def on_message(cli: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage):
    try:
        raw = msg.payload.decode("utf-8")
        log.info(f"Received message on {msg.topic}: {raw}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.error("Invalid JSON received")
            return

        error = OllamaRequestValidator.validate(data)
        if error:
            log.error(f"Validation error: {error}")
            return

        payload = {
            "model": data["model"],
            "system": data["system"],
            "prompt": data["user"],
            "stream": False,
        }

        # Add optional parameters if present
        if "temperature" in data:
            payload["temperature"] = data["temperature"]
        if "top_p" in data:
            payload["top_p"] = data["top_p"]
        if "top_k" in data:
            payload["top_k"] = data["top_k"]

        log.info(f"Calling Ollama for request_id={data['request_id']}")

        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()

        log.info(f"Ollama response for {data['request_id']}: {result}")

        # You can publish the result back if needed
        # client.publish(RESPONSE_TOPIC, json.dumps(...))

    except Exception as e:
        log.error(f"Error handling message: {e}")





client = mqtt.Client(client_id="ollama", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
# client.max_queued_messages_set(5)
# client.will_set(MQTT_TOPIC, json.dumps({"heartbeat": 0, "status": "offline"}), qos=1, retain=True)
client.reconnect_delay_set(min_delay=1, max_delay=30)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

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
# Helper functions
# ------------------------------

class OllamaRequestValidator:
    REQUIRED_FIELDS = {
        "request_id": str,
        "model": str,
        "system": str,
        "user": str,
    }

    OPTIONAL_FIELDS = {
        "temperature": (int, float),
        "top_p": (int, float),
        "top_k": int,
    }

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Optional[str]:
        # Check required fields
        for field, expected_type in cls.REQUIRED_FIELDS.items():
            if field not in data:
                return f"Missing required field: {field}"
            if not isinstance(data[field], expected_type):
                return f"Field '{field}' must be of type {expected_type.__name__}"

        # Check optional fields
        for field, expected_type in cls.OPTIONAL_FIELDS.items():
            if field in data and not isinstance(data[field], expected_type):
                if isinstance(expected_type, tuple):
                    expected = ", ".join(t.__name__ for t in expected_type)
                else:
                    expected = expected_type.__name__
                return f"Field '{field}' must be of type {expected}"

        return None  # valid

    @classmethod
    def extract(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        # Returns a cleaned dict with only known fields
        result = {k: data[k] for k in cls.REQUIRED_FIELDS if k in data}
        for k in cls.OPTIONAL_FIELDS:
            if k in data:
                result[k] = data[k]
        return result









def publish_telemetry(jetson: jtop) -> None:
    try:

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
            # publish_telemetry(jetsonTop)
            # time.sleep(PUBLISH_INTERVAL)

except SystemExit:
    log.debug("SystemExit received, stopping main loop.")

except Exception as e:
    log.error(f"Error in main loop: {e}")

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
