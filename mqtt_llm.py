#!/usr/bin/env python3

import argparse
import http.client
import itertools
import json
import logging
import queue
import signal
import socket
import threading
import time
from enum import IntEnum
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
from pydantic import BaseModel, Field, ValidationError

# ------------------------------
# Command-line arguments
# ------------------------------

def _port_type(value: str) -> int:
    ivalue = int(value)
    if not (1 <= ivalue <= 65535):
        raise argparse.ArgumentTypeError(f"must be between 1 and 65535, got {ivalue}")
    return ivalue

def _positive_float(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {fvalue}")
    return fvalue

def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {ivalue}")
    return ivalue

logChoices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
parser = argparse.ArgumentParser(description="MQTT interface to Ollama")
parser.add_argument("--broker", required=True, help="MQTT broker IP or hostname")
parser.add_argument("--port", type=_port_type, default=1883, help="MQTT broker port (default 1883)")
parser.add_argument("--username", help="MQTT username")
parser.add_argument("--credpath", help="LoadCredential path for MQTT password")
parser.add_argument("--topic", default="orin/ollama", help="MQTT topic to publish and receive")
parser.add_argument("--ollama-host", default="localhost", help="Ollama host address (default: localhost)")
parser.add_argument("--ollama-port", type=_port_type, default="11434", help="Ollama host address (default: 11434)")
parser.add_argument("--connect-timeout", type=_positive_float, default=5.0, help="Max seconds to establish Ollama TCP connection")
parser.add_argument("--stream-timeout", type=_positive_float, default=120.0, help="Max seconds of silence between streamed chunks")
parser.add_argument("--shutdown-timeout", type=_positive_float, default=5.0, help="Max seconds to full service shutdown")
parser.add_argument("--interval", type=_positive_int, default=10, help="Status message publish interval in seconds")
parser.add_argument("--loglevel", default="INFO", choices=logChoices, help="Logging level")
args = parser.parse_args()

MQTT_BROKER: str = args.broker
MQTT_PORT: int = args.port
MQTT_TOPIC: str = args.topic
MQTT_STATUS_TOPIC: str = f"{MQTT_TOPIC}/status"
USERNAME: str | None = args.username
CRED_PATH = Path(args.credpath) if args.credpath else None
OLLAMA_HOST: str = args.ollama_host
OLLAMA_PORT: int = args.ollama_port
OLLAMA_PATH: str = "/api/generate"
OLLAMA_CONNECT_TIMEOUT: float = args.connect_timeout
OLLAMA_STREAM_IDLE_TIMEOUT: float = args.stream_timeout
SHUTDOWN_TIMEOUT: float = args.shutdown_timeout
STATUS_INTERVAL: int = args.interval

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
    temperature: float | None = Field(default=None, strict=True)
    top_p: float | None = Field(default=None, strict=True)
    top_k: int | None = Field(default=None, strict=True)

    def to_ollama_options(self) -> dict[str, Any]:
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

def handle_request_error(request_id: str | None, code: ErrorCode, message: Any) -> None:
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
connected_event = threading.Event()
CONNECT_RETRY_LOG_INTERVAL = 5.0  # how often to log while waiting for the initial connection
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
        self._sock: socket.socket | None = None

    @property
    def conn(self) -> http.client.HTTPConnection:
        return self._conn

    @property
    def sock(self) -> socket.socket | None:
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

active_responses: dict[str, ActiveGeneration] = {}

class GenerationCancelled(Exception):
    """Raised when an in-flight Ollama generation is aborted due to shutdown."""

# ------------------------------
# Queue status tracking
# ------------------------------
# request_id is caller-supplied and must be treated as an arbitrary,
# opaque label: it's never validated for uniqueness (or non-emptiness
# beyond being a string), so nothing internal may use it as a lookup
# key - two, or two hundred, queued requests can legitimately share the
# same request_id (or all be ""). queue_status is therefore keyed by
# _queue_seq (a private, always-unique counter assigned at enqueue
# time), with each entry's own "request_id" field carried purely as a
# label to report back, never as an identity. Insertion order into this
# dict == processing order (queue_status is a plain dict, and
# request_queue is FIFO), so this also doubles as the queue's own order
# record - no separate structure needed to answer "what order will
# these process in".
_queue_seq_counter = itertools.count()
queue_status_lock = threading.Lock()
queue_status: dict[int, dict[str, Any]] = {}

def update_queue_progress(seq: int, response_chars: int, thinking_chars: int) -> None:
    with queue_status_lock:
        entry = queue_status.get(seq)
        if entry is not None:  # already removed (e.g. shutdown drained it) - nothing to update
            entry["response_chars"] = response_chars
            entry["thinking_chars"] = thinking_chars

def build_queue_snapshot() -> list[dict[str, Any]]:
    """The queue's essence for a status subscriber: IDs in the order
    they'll be processed, each with independent response/thinking
    progress counters. response_chars follows -1 queued but not yet
    started, 0 started with nothing generated yet, otherwise the
    current character count. thinking_chars follows the same -1/0/N
    shape, but -1 also covers "no thinking activity has been observed
    for this request yet" - since not every model streams a "thinking"
    field, it only leaves -1 once one actually appears (see the
    thinking_started tracking in stream_ollama_generate)."""
    with queue_status_lock:
        return [
            {
                "request_id": entry["request_id"],
                "response_chars": entry["response_chars"],
                "thinking_chars": entry["thinking_chars"],
            }
            for entry in queue_status.values()
        ]

def build_status_payload(online: bool) -> dict[str, Any]:
    if not online:
        return {"status": "offline", "heartbeat": None, "queue": None}
    return {
        "status": "online",
        "heartbeat": int(time.time()),
        "queue": build_queue_snapshot(),
    }

# Serializes build_status_payload()+cli.publish() as one unit across
# threads (on_message, ollama_worker, status_publisher all call this).
# Without it, two concurrent calls could interleave so the one that
# snapshotted queue_status *first* is still the one whose publish()
# reaches the broker *last* - leaving a stale, smaller retained queue
# in place of a fresher, larger one it should have superseded.
status_publish_lock = threading.Lock()

def publish_status(cli: mqtt.Client, online: bool = True) -> None:
    with status_publish_lock:
        payload = build_status_payload(online)
        try:
            info = cli.publish(MQTT_STATUS_TOPIC, json.dumps(payload), qos=1, retain=True)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                log.warning(f"Status publish returned error code: {info.rc}")
        except Exception as e:
            log.error(f"Failed to publish status: {e}")

# ------------------------------
# MQTT setup
# ------------------------------

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
            sub_topic = f"{MQTT_TOPIC}/request"
            log.info(f"Connected to MQTT broker, listening to '{sub_topic}'.")
            cli.subscribe(sub_topic, qos=1)
            publish_status(cli)
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

def wait_for_mqtt_connection() -> None:
    """Block until the MQTT client is genuinely connected, or shutdown
    is requested.

    client.connect() only sends the CONNECT packet; the CONNACK that
    flips is_connected() to True is processed asynchronously by the
    loop thread. Without waiting for it here, a shutdown signal arriving
    before that CONNACK lands could race the shutdown block's
    is_connected() check: it could read False (skipping the explicit
    offline publish) while the CONNACK then lands moments later and lets
    the eventual disconnect() go out clean - which cancels LWT delivery
    too. Net effect: no offline announcement at all. handle_exit() also
    sets connected_event, so a shutdown request wakes this promptly
    instead of waiting out the full log interval.
    """
    while not client.is_connected() and not shutdown_event.is_set():
        connected_event.clear()
        if client.is_connected() or shutdown_event.is_set():
            continue  # settled in the instant between the check above and clear()
        if not connected_event.wait(timeout=CONNECT_RETRY_LOG_INTERVAL):
            log.warning(
                f"Still waiting for MQTT connection after {CONNECT_RETRY_LOG_INTERVAL}s; "
                "retrying in the background..."
            )

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

        seq = next(_queue_seq_counter)  # unique tracking key - request_id may repeat or be ""
        with queue_status_lock:
            queue_status[seq] = {"request_id": request.request_id, "response_chars": -1, "thinking_chars": -1}
        try:
            request_queue.put((seq, request))
            log.debug(
                f"Queued request_id={request.request_id} "
                f"(queue depth={request_queue.qsize()})"
            )
        except Exception:
            # Roll back the tracking entry so the status published below
            # accurately shows this request never made it into the queue.
            with queue_status_lock:
                queue_status.pop(seq, None)
            raise
        finally:
            # Published immediately (not just on the periodic interval) so
            # the requestor can see their ID land in the queue right away,
            # or notice its absence if something above removed it again.
            publish_status(cli)
    except Exception as e:
        log.error(f"Unhandled error in on_message: {e}")

# ------------------------------
# Ollama worker
# ------------------------------

def build_ollama_payload(request: OllamaRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "system": request.system,
        "prompt": request.user,
        "stream": True,
    }

    options = request.to_ollama_options()
    if options:
        payload["options"] = options

    return payload

def stream_ollama_generate(request: OllamaRequest, seq: int | None = None) -> dict[str, Any]:
    """seq is the caller's queue_status tracking key (None if the caller
    isn't tracking queue progress for this call, e.g. direct unit tests) -
    deliberately not request.request_id, which may repeat across requests
    or be empty and so can't identify anything on its own."""
    payload = build_ollama_payload(request)
    body = json.dumps(payload).encode("utf-8")

    response_parts = []
    thinking_parts = []  # some reasoning models stream a separate "thinking" field
    thinking_started = False  # latches once a "thinking" key is ever observed - not every model sends one
    final_meta: dict[str, Any] = {}

    start = time.monotonic()
    last_progress_log = start
    chunk_count = 0

    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=OLLAMA_CONNECT_TIMEOUT)
    active_call = ActiveGeneration(conn)

    with active_responses_lock:
        # Checked under the same lock the shutdown block uses to snapshot
        # and interrupt active_responses, so there's no window where a
        # request dequeued just as shutdown begins could register after
        # that snapshot was taken and run uninterrupted to completion.
        if shutdown_event.is_set():
            raise GenerationCancelled()
        active_responses[request.request_id] = active_call

    try:
        try:
            active_call.connect()
            assert active_call.sock is not None, "conn.connect() succeeded without setting a socket"
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
                if not isinstance(chunk, dict):
                    # Guards chunk.get(...)/final_meta.items() below - a
                    # non-object line is a protocol violation, not
                    # something to fail on with a raw AttributeError.
                    raise RuntimeError(f"Ollama sent a non-object chunk: {chunk!r}")
                chunk_count += 1

                if chunk.get("response"):
                    response_parts.append(chunk["response"])
                if "thinking" in chunk:
                    thinking_started = True  # this model streams a thinking field, even if this chunk's is empty
                if chunk.get("thinking"):
                    thinking_parts.append(chunk["thinking"])

                if seq is not None:
                    update_queue_progress(
                        seq,
                        response_chars=sum(len(p) for p in response_parts),
                        thinking_chars=sum(len(p) for p in thinking_parts) if thinking_started else -1,
                    )

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
        item = request_queue.get()  # blocks / waits if empty

        if item is SHUTDOWN_SENTINEL:
            log.info("Shutdown sentinel received, exiting worker loop")
            request_queue.task_done()
            break

        seq, request = item  # seq is the queue_status tracking key, not request.request_id

        if shutdown_event.is_set():  # drain queue on shutdown
            log.info(f"Discarding queued request_id={request.request_id} due to shutdown")
            with queue_status_lock:
                queue_status.pop(seq, None)
            request_queue.task_done()
            continue

        with queue_status_lock:
            entry = queue_status.get(seq)
            if entry is not None:  # dequeued: response generation started, nothing yet
                entry["response_chars"] = 0
                # thinking_chars deliberately stays at -1 here: whether this
                # request's model streams a "thinking" field at all is only
                # known once (if ever) one actually shows up in a chunk, in
                # stream_ollama_generate's thinking_started tracking below.

        try:
            log.info(f"Calling Ollama for request_id={request.request_id}")
            result = stream_ollama_generate(request, seq)
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
            with queue_status_lock:
                queue_status.pop(seq, None)
            if not shutdown_event.is_set():
                # Shutdown publishes its own final offline status right
                # after this; skip the redundant online one mid-teardown.
                publish_status(client)
            request_queue.task_done()

    log.info("Worker thread stopped")


# ------------------------------
# MQTT client instantiation
# ------------------------------

client = mqtt.Client(client_id="orin_ollama", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.max_queued_messages_set(5)
client.will_set(MQTT_STATUS_TOPIC, json.dumps(build_status_payload(online=False)), qos=1, retain=True)
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

try:
    client.loop_start()  # background thread handles the network loop and reconnects
except Exception as e:
    log.error(f"Failed to start MQTT network loop: {e}")
    raise  # let systemd restart

# ------------------------------
# Shutdown handling
# ------------------------------

def handle_exit(signum: int, frame: Any | None) -> None:
    log.info(f"Received signal {signum}, shutting down...")
    shutdown_event.set()
    connected_event.set()  # wake wait_for_mqtt_connection() promptly too, if it's blocked there

# Registered here, before wait_for_mqtt_connection() is ever called, so a
# signal arriving while still waiting for the initial connection can
# interrupt that wait too, not just the final shutdown_event.wait() below.
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

wait_for_mqtt_connection()

# ------------------------------
# Worker thread startup
# ------------------------------
worker_thread = threading.Thread(target=ollama_worker, name="ollama-worker", daemon=True)
try:
    worker_thread.start()  # running as daemon makes OS cancel the thread as last resort on timeout
except Exception as e:
    log.error(f"Failed to start worker thread: {e}")
    raise  # let systemd restart

# ------------------------------
# Status publisher thread startup
# ------------------------------

def status_publisher() -> None:
    log.info(f"Status publisher thread started (interval={STATUS_INTERVAL}s)")
    while not shutdown_event.wait(timeout=STATUS_INTERVAL):
        publish_status(client)
    log.info("Status publisher thread stopped")

status_thread = threading.Thread(target=status_publisher, name="status-publisher", daemon=True)
try:
    status_thread.start()
except Exception as e:
    log.error(f"Failed to start status publisher thread: {e}")
    raise  # let systemd restart

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

    status_thread.join(timeout=SHUTDOWN_TIMEOUT)

    if status_thread.is_alive():
        log.warning(
            f"Status publisher thread did not finish within {SHUTDOWN_TIMEOUT}s; "
            "proceeding with shutdown anyway - the thread is a daemon and will "
            "be discarded on process exit."
        )
    else:
        log.debug("Status publisher thread exited cleanly.")

    try:
        if client.is_connected():
            info = client.publish(MQTT_STATUS_TOPIC, json.dumps(build_status_payload(online=False)), qos=1, retain=True)
            info.wait_for_publish(timeout=2.0)  # ensure the offline status is sent before disconnecting
            log.debug("Offline status published.")
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        log.warning(f"Error while stopping MQTT client: {e}")
    log.info("Shutdown complete.")
