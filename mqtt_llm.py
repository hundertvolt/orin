#!/usr/bin/env python3

import argparse
import http.client
import json
import logging
import queue
import signal
import socket
import threading
import time

import paho.mqtt.client as mqtt
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

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
parser.add_argument("--connect-timeout", type=float, default=5.0, help="Max seconds to establish Ollama TCP connection")
parser.add_argument("--stream-timeout", type=float, default=120.0, help="Max seconds of silence between streamed chunks")
parser.add_argument("--shutdown-timeout", type=float, default=5.0, help="Max seconds to full service shutdown")
parser.add_argument("--loglevel", default="INFO", choices=logChoices, help="Logging level")
args = parser.parse_args()

MQTT_BROKER: str = args.broker
MQTT_PORT: int = args.port
MQTT_TOPIC: str = args.topic
MQTT_STATUS_TOPIC: str = f"{MQTT_TOPIC}/status"
USERNAME: Optional[str] = args.username
CRED_PATH = Path(args.credpath) if args.credpath else None
OLLAMA_HOST: str = args.ollama_host
OLLAMA_PORT: int = args.ollama_port
OLLAMA_PATH: str = "/api/generate"
OLLAMA_CONNECT_TIMEOUT: float = args.connect_timeout
OLLAMA_STREAM_IDLE_TIMEOUT: float = args.stream_timeout
SHUTDOWN_TIMEOUT: float = args.shutdown_timeout

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    level=getattr(logging, args.loglevel.upper(), logging.INFO),
    format="[%(levelname)s] [%(threadName)s] [%(name)s] %(message)s"
)

log = logging.getLogger("ollama_mqtt")
threading.current_thread().name = "main"

# ------------------------------
# Request schema and error codes
# ------------------------------

class ErrorCode(IntEnum):
    OK = 0
    INVALID_JSON = 1
    VALIDATION_ERROR = 2
    OLLAMA_ERROR = 3

class OllamaRequest(BaseModel):
    request_id: str = Field(strict=True)
    model: str = Field(strict=True)
    system: str = Field(strict=True)
    user: str = Field(strict=True)
    temperature: Optional[float] = Field(default=None, strict=True)
    top_p: Optional[float] = Field(default=None, strict=True)
    top_k: Optional[int] = Field(default=None, strict=True)

    def to_ollama_options(self) -> Dict[str, Any]:
        return self.model_dump(include={"temperature", "top_p", "top_k"}, exclude_none=True)

def format_validation_error(exc: ValidationError) -> str:
    # Collapses pydantic's structured error list into a single string,
    # e.g. "temperature: value is not a valid float; user: field required"
    parts = []
    for err in exc.errors(include_url=False, include_context=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)

def publish_response(
    request_id: str, message: str, error_code: ErrorCode, error_message: str
) -> None:
    payload = {
        "request_id": request_id,
        "message": message,
        "error_code": int(error_code),
        "error_message": error_message,
    }
    try:
        info = client.publish(f"{MQTT_TOPIC}/response", json.dumps(payload), qos=1)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning(f"Publish returned error code: {info.rc} for request_id={request_id}")
    except Exception as e:
        log.error(f"Failed to publish response for request_id={request_id}: {e}")

def handle_request_error(request_id: Optional[str], code: ErrorCode, message: Any) -> None:
    error_message = str(message)
    log.error(f"[error_code={int(code)}] request_id={request_id}: {error_message}")

    if request_id is None:
        log.debug("No request_id available; skipping MQTT error response.")
        return

    publish_response(request_id, message="", error_code=code, error_message=error_message)

# ------------------------------
# Request queue
# ------------------------------
request_queue: "queue.Queue[Any]" = queue.Queue()
SHUTDOWN_SENTINEL = object()
shutdown_event = threading.Event()
active_responses_lock = threading.Lock()

class ActiveGeneration:
    """Tracks an in-flight Ollama call so shutdown can interrupt it.

    http.client's getresponse() closes and nulls HTTPConnection.sock as
    soon as it reads a response with no Content-Length (true for our
    streamed NDJSON responses) - "passing" the live socket to the
    response object instead. So conn.sock can't be relied on to reach
    the connection once streaming has started. We capture the raw
    socket reference right after connect(), before getresponse() can
    null it, and shut that down directly - the underlying socket object
    stays valid (only conn's reference to it is cleared) and its
    shutdown() call still interrupts a thread blocked reading it.
    """

    def __init__(self, conn: http.client.HTTPConnection) -> None:
        self._conn = conn
        self._sock: Optional[socket.socket] = None

    @property
    def conn(self) -> http.client.HTTPConnection:
        return self._conn

    @property
    def sock(self) -> Optional[socket.socket]:
        return self._sock

    def connect(self) -> None:
        """Open the connection and capture its socket, before getresponse() can null it."""
        self._conn.connect()
        self._sock = self._conn.sock

    def shutdown(self) -> None:
        """Interrupt a thread blocked reading the captured socket, if any."""
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already closed/disconnected - nothing to unblock

    def close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass  # already closed/disconnected - nothing to clean up

active_responses: Dict[str, ActiveGeneration] = {}

class GenerationCancelled(Exception):
    """Raised when an in-flight Ollama generation is aborted due to shutdown."""

# ------------------------------
# MQTT setup
# ------------------------------

def on_connect(cli: mqtt.Client, userdata: Any, flags: Dict[str, int], reason_code: int, properties: Any) -> None:
    # paho re-raises exceptions escaping this callback, which kills its network
    # loop thread for good, so every path out of here must be caught locally.
    try:
        if reason_code == 0:
            sub_topic = f"{MQTT_TOPIC}/request"
            log.info(f"Connected to MQTT broker, listening to '{sub_topic}'.")
            cli.subscribe(sub_topic, qos=1)
            cli.publish(MQTT_STATUS_TOPIC, json.dumps({"status": "online"}), qos=1, retain=True)
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

def on_message(cli: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
    try:
        try:
            raw = msg.payload.decode("utf-8")
        except Exception as e:
            log.error(f"Failed to decode message payload: {e}")
            return

        log.info(f"Received message on {msg.topic}: {raw}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            handle_request_error(request_id=None, code=ErrorCode.INVALID_JSON, message=str(e))
            return

        fallback_id = data.get("request_id") if isinstance(data, dict) else None

        try:
            request = OllamaRequest.model_validate(data)
        except ValidationError as e:
            handle_request_error(
                request_id=fallback_id,
                code=ErrorCode.VALIDATION_ERROR,
                message=format_validation_error(e),
            )
            return

        request_queue.put(request)
        log.debug(
            f"Queued request_id={request.request_id} "
            f"(queue depth={request_queue.qsize()})"
        )
    except Exception as e:
        log.error(f"Unhandled error in on_message: {e}")

# ------------------------------
# Ollama worker
# ------------------------------

def build_ollama_payload(request: OllamaRequest) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": request.model,
        "system": request.system,
        "prompt": request.user,
        "stream": True,
    }

    options = request.to_ollama_options()
    if options:
        payload["options"] = options

    return payload

def stream_ollama_generate(request: OllamaRequest) -> Dict[str, Any]:
    payload = build_ollama_payload(request)
    body = json.dumps(payload).encode("utf-8")

    response_parts = []
    thinking_parts = []  # some reasoning models stream a separate "thinking" field
    final_meta: Dict[str, Any] = {}

    start = time.monotonic()
    last_progress_log = start
    chunk_count = 0

    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=OLLAMA_CONNECT_TIMEOUT)
    active_call = ActiveGeneration(conn)

    with active_responses_lock:
        active_responses[request.request_id] = active_call

    try:
        try:
            active_call.connect()
            active_call.sock.settimeout(OLLAMA_STREAM_IDLE_TIMEOUT)  # switch to idle timeout once connected

            active_call.conn.request("POST", OLLAMA_PATH, body=body, headers={"Content-Type": "application/json"})
            resp = active_call.conn.getresponse()

            if resp.status != 200:
                error_body = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama returned HTTP {resp.status}: {error_body}")

            stream_completed = False
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                chunk_count += 1

                if chunk.get("response"):
                    response_parts.append(chunk["response"])
                if chunk.get("thinking"):
                    thinking_parts.append(chunk["thinking"])

                now = time.monotonic()
                if now - last_progress_log >= 5.0:
                    log.info(
                        f"Still generating request_id={request.request_id}: "
                        f"{chunk_count} chunks, "
                        f"{sum(len(p) for p in response_parts)} chars so far "
                        f"({now - start:.1f}s elapsed)"
                    )
                    last_progress_log = now

                if chunk.get("done"):
                    final_meta = chunk
                    stream_completed = True
                    break

            if not stream_completed:
                # The connection ended (closed/reset/EOF) before Ollama sent a
                # final "done" chunk - the caller must not mistake this
                # truncated output for a complete response.
                raise RuntimeError(
                    f"Ollama stream ended unexpectedly after {chunk_count} chunk(s) "
                    "without a final 'done' marker"
                )
        finally:
            with active_responses_lock:
                active_responses.pop(request.request_id, None)
            active_call.close()

    except Exception:  # catch all exceptions and explicitly classify for deliberate cancellation
        if shutdown_event.is_set():
            raise GenerationCancelled() from None
        raise

    return {
        "response": "".join(response_parts),
        "thinking": "".join(thinking_parts) if thinking_parts else None,
        **{k: v for k, v in final_meta.items() if k not in ("response", "thinking")},
    }


def ollama_worker() -> None:
    log.info("Worker thread started")
    while True:
        request = request_queue.get()  # blocks / waits if empty

        if request is SHUTDOWN_SENTINEL:
            log.info("Shutdown sentinel received, exiting worker loop")
            request_queue.task_done()
            break

        if shutdown_event.is_set():  # drain queue on shutdown
            log.info(f"Discarding queued request_id={request.request_id} due to shutdown")
            request_queue.task_done()
            continue

        try:
            log.info(f"Calling Ollama for request_id={request.request_id}")
            result = stream_ollama_generate(request)
            log.info(f"Ollama response for {request.request_id}: {result}")

            publish_response(
                request.request_id,
                message=result.get("response", ""),
                error_code=ErrorCode.OK,
                error_message="OK",
            )

        except GenerationCancelled:
            log.info(f"Generation for request_id={request.request_id} cancelled due to shutdown")
        except (OSError, http.client.HTTPException, RuntimeError) as e:
            handle_request_error(request.request_id, ErrorCode.OLLAMA_ERROR, str(e))
        except Exception as e:
            handle_request_error(request.request_id, ErrorCode.OLLAMA_ERROR, f"unexpected error: {e}")
        finally:
            request_queue.task_done()

    log.info("Worker thread stopped")


# ------------------------------
# MQTT client instantiation
# ------------------------------

client = mqtt.Client(client_id="orin_ollama", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.max_queued_messages_set(5)
client.will_set(MQTT_STATUS_TOPIC, json.dumps({"status": "offline"}), qos=1, retain=True)
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

client.loop_start()  # background thread handles the network loop and reconnects

# ------------------------------
# Worker thread startup
# ------------------------------
worker_thread = threading.Thread(target=ollama_worker, name="ollama-worker", daemon=True)
worker_thread.start()  # running as daemon makes OS cancel the thread as last resort on timeout

# ------------------------------
# Shutdown handling
# ------------------------------

def handle_exit(signum: int, frame: Optional[Any]) -> None:
    log.info(f"Received signal {signum}, shutting down...")
    shutdown_event.set()

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

log.info(f"Ollama MQTT bridge started, listening on '{MQTT_TOPIC}/request'.")

try:
    shutdown_event.wait() # Main thread parks here
finally:
    log.info("Stopping worker thread and MQTT client...")
    request_queue.put(SHUTDOWN_SENTINEL)

    with active_responses_lock:
        for call in list(active_responses.values()):
            call.shutdown()
            call.close()

    worker_thread.join(timeout=SHUTDOWN_TIMEOUT)

    if worker_thread.is_alive():
        log.warning(
            f"Worker thread did not finish within {SHUTDOWN_TIMEOUT}s "
            "even after cancelling; proceeding with shutdown anyway - "
            "the thread is a daemon and will be discarded on process exit."
        )
    else:
        log.debug("Worker thread exited cleanly.")

    try:
        if client.is_connected():
            info = client.publish(MQTT_STATUS_TOPIC, json.dumps({"status": "offline"}), qos=1, retain=True)
            info.wait_for_publish(timeout=2.0)  # ensure the offline status is sent before disconnecting
            log.debug("Offline status published.")
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        log.warning(f"Error while stopping MQTT client: {e}")
    log.info("Shutdown complete.")
