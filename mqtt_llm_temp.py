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

from pydantic import BaseModel, ValidationError

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
parser.add_argument("--loglevel", default="INFO", choices=logChoices, help="Logging level")
parser.add_argument(
    "--connect-timeout",
    type=float,
    default=5.0,
    help="Max seconds to establish the TCP connection to Ollama (default 5)",
)
parser.add_argument(
    "--stream-idle-timeout",
    type=float,
    default=120.0,
    help=(
        "Max seconds of silence between streamed chunks before considering "
        "the generation stalled (default 120). This is NOT a total-duration "
        "limit - as long as new tokens keep arriving, a generation can run "
        "far longer than this without triggering it."
    ),
)
parser.add_argument(
    "--shutdown-timeout",
    type=float,
    default=5.0,
    help=(
        "Max seconds to wait for the worker thread to notice a cancelled "
        "generation and unwind after shutdown is requested (default 5). "
        "This no longer bounds how long an Ollama call itself may run - "
        "in-flight connections are force-closed on shutdown instead."
    ),
)

args = parser.parse_args()

MQTT_BROKER: str = args.broker
MQTT_PORT: int = args.port
MQTT_TOPIC: str = args.topic
USERNAME: Optional[str] = args.username
CRED_PATH = Path(args.credpath) if args.credpath else None
OLLAMA_HOST: str = args.ollama_host
OLLAMA_PORT: int = args.ollama_port
OLLAMA_PATH: str = "/api/generate"
OLLAMA_CONNECT_TIMEOUT: float = args.connect_timeout
OLLAMA_STREAM_IDLE_TIMEOUT: float = args.stream_idle_timeout
SHUTDOWN_TIMEOUT: float = args.shutdown_timeout
RESPONSE_TOPIC: str = f"{MQTT_TOPIC}/response"

# ------------------------------
# Logging setup
# ------------------------------

# %(threadName)s is included deliberately: this script runs the MQTT
# network loop, the Ollama worker, and the main thread concurrently -
# thread names in every log line make it possible to see at a glance
# which thread did what, and to confirm none of them are blocking another.
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
    request_id: str
    model: str
    system: str
    user: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    def to_ollama_options(self) -> Dict[str, Any]:
        # Only the sampling params that were actually set - pydantic
        # keeps required/optional and typing in one place, so this
        # never drifts out of sync with validation the way two
        # hand-maintained field dicts could.
        return self.model_dump(
            include={"temperature", "top_p", "top_k"}, exclude_none=True
        )


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
        info = client.publish(RESPONSE_TOPIC, json.dumps(payload), qos=1)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning(
                f"Publish returned error code: {info.rc} for request_id={request_id}"
            )
    except Exception as e:
        log.error(f"Failed to publish response for request_id={request_id}: {e}")


def handle_request_error(request_id: Optional[str], code: ErrorCode, message: Any) -> None:
    error_message = str(message)
    log.error(f"[error_code={int(code)}] request_id={request_id}: {error_message}")

    if request_id is None:
        # Nothing to correlate a response to - e.g. invalid JSON, where
        # parsing failed before any request_id could even be extracted
        # from the payload. Nothing useful to publish in that case.
        log.debug("No request_id available; skipping MQTT error response.")
        return

    publish_response(request_id, message="", error_code=code, error_message=error_message)

# ------------------------------
# Request queue
# ------------------------------

# Validated requests land here from on_message (paho's network thread).
# A separate worker thread drains it one at a time, so a slow/blocking
# Ollama call never stalls MQTT keepalive or the next incoming request.
#
# SHUTDOWN_SENTINEL is a unique object (not None, not any real request)
# put onto this same queue to tell the worker thread to stop pulling new
# work and exit its loop - the standard "poison pill" pattern for a
# queue.Queue-based worker.
request_queue: "queue.Queue[Any]" = queue.Queue()
SHUTDOWN_SENTINEL = object()

# Set by the signal handler; checked after catching an exception from a
# streamed Ollama call to distinguish "we cancelled this on purpose" from
# a genuine network/server failure - see stream_ollama_generate() below
# for why this has to be a post-hoc check rather than a specific except
# clause.
shutdown_event = threading.Event()

# The HTTPConnection for whichever request is currently in flight, keyed by
# request_id, so the shutdown sequence can force-close its socket and
# unblock the worker immediately instead of waiting on --stream-idle-timeout
# to notice on its own.
#
# This is deliberately a raw http.client.HTTPConnection rather than a
# requests.Response: requests/urllib3 only hand you a Response object once
# the status line has already been read - which is exactly the phase we
# need to interrupt, since Ollama sends nothing at all (not even headers)
# until the first token is generated, and that can take 30-80s on a cold
# model load. http.client gives us the raw socket back from .connect(),
# before anything is sent or read, so it can be registered - and
# cancelled - before that wait even begins.
active_responses: Dict[str, http.client.HTTPConnection] = {}
active_responses_lock = threading.Lock()


class GenerationCancelled(Exception):
    """Raised when an in-flight Ollama generation is aborted due to shutdown."""


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

def on_message(cli: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
    # Runs on paho's network thread. Everything here is cheap (decode,
    # JSON parse, pydantic validation, a queue.put) - there is no network
    # call to Ollama on this thread, so incoming MQTT traffic (keepalive
    # pings, further requests) is never held up waiting for a generation
    # to finish.
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

    # Best-effort request_id for correlation, even if the rest of the
    # payload fails validation below - lets a future error response
    # still reference the caller's id.
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


# ------------------------------
# Ollama worker
# ------------------------------

def build_ollama_payload(request: OllamaRequest) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": request.model,
        "system": request.system,
        "prompt": request.user,
        # With stream=False, Ollama runs the entire generation server-side
        # and sends nothing over the wire until it's completely done - for
        # a thinking model or a long output, that can be many minutes of
        # total silence, and a client-side timeout on that call is really
        # just a guess at the worst-case total generation time. Streaming
        # sends one JSON chunk per token as it's produced, which turns the
        # timeout question into "how long since the last chunk", not "how
        # long in total" - a generation can run indefinitely as long as
        # tokens keep arriving.
        "stream": True,
    }

    # Sampling parameters belong nested under "options" for /api/generate -
    # Ollama silently ignores them at the top level.
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

    # timeout= here bounds only the TCP connect - deliberately short and
    # separate from the idle timeout below, same split --connect-timeout /
    # --stream-idle-timeout previously expressed via requests' timeout tuple.
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=OLLAMA_CONNECT_TIMEOUT)

    # Registered immediately, before connect() even runs: this is what makes
    # the whole wait - connect, header wait, and between-chunk wait alike -
    # cancellable from the shutdown sequence, instead of only the streaming
    # phase as before. See the active_responses definition above for why.
    with active_responses_lock:
        active_responses[request.request_id] = conn

    try:
        try:
            conn.connect()

            # Now that we're connected, switch the socket from the connect
            # timeout to the idle timeout - this single socket-level timeout
            # covers both "waiting for the first byte of the response" (the
            # cold-start delay) and "waiting between streamed chunks",
            # matching what --stream-idle-timeout is documented to mean.
            conn.sock.settimeout(OLLAMA_STREAM_IDLE_TIMEOUT)

            conn.request(
                "POST",
                OLLAMA_PATH,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()

            if resp.status != 200:
                error_body = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama returned HTTP {resp.status}: {error_body}")

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
                    break
        finally:
            with active_responses_lock:
                active_responses.pop(request.request_id, None)
            conn.close()

    except Exception:
        # A forced shutdown/close() from the shutdown sequence below
        # surfaces as whatever error the socket layer or http.client raises
        # when the connection disappears mid-wait - typically an OSError
        # (e.g. "Bad file descriptor") or http.client.HTTPException,
        # depending on exactly which blocking call it interrupts. There is
        # no single reliable exception type to catch for "this was
        # cancelled", so instead we catch broadly here and classify after
        # the fact using our own shutdown_event flag.
        if shutdown_event.is_set():
            raise GenerationCancelled() from None
        raise

    return {
        "response": "".join(response_parts),
        "thinking": "".join(thinking_parts) if thinking_parts else None,
        **{k: v for k, v in final_meta.items() if k not in ("response", "thinking")},
    }


def ollama_worker() -> None:
    # Runs on its own thread. requests.post() blocks this thread while
    # waiting on the socket, but that block releases the GIL for the
    # entire wait - CPython always releases the GIL around blocking
    # syscalls (socket recv/send, queue.get, etc). The actual LLM
    # inference happens inside the separate `ollama serve` process, not
    # in this Python process at all, so there is no CPU contention with
    # the MQTT network thread either way - this thread is doing nothing
    # but waiting on I/O the whole time it's "busy".
    log.info("Worker thread started")
    while True:
        request = request_queue.get()

        if request is SHUTDOWN_SENTINEL:
            log.info("Shutdown sentinel received, exiting worker loop")
            request_queue.task_done()
            break

        if shutdown_event.is_set():
            # The sentinel is appended to the end of the queue, so it does
            # not jump ahead of requests that were already queued when
            # shutdown began - without this check, the worker would keep
            # starting fresh Ollama calls for that backlog before ever
            # reaching the sentinel. Once shutdown_event is set, nothing
            # dequeued gets processed - it's discarded so the worker can
            # drain straight through to the sentinel and exit promptly.
            log.info(
                f"Discarding queued request_id={request.request_id} "
                "due to shutdown"
            )
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
            log.info(
                f"Generation for request_id={request.request_id} "
                "cancelled due to shutdown"
            )
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

# daemon=True is a last-resort safety net, not the primary shutdown
# mechanism: the sentinel plus the forced connection close (further
# below) are what ask the worker to stop and unblock immediately.
# daemon=True only matters if the worker is somehow still stuck past
# SHUTDOWN_TIMEOUT even after that - in that case the process still
# exits instead of hanging forever, and the OS reclaims the thread.
worker_thread = threading.Thread(target=ollama_worker, name="ollama-worker", daemon=True)
worker_thread.start()

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
    # Main thread parks here - shutdown_event.wait() blocks on a
    # condition variable (GIL released) until a signal handler sets it,
    # so it costs nothing while idle, exactly like ollama_worker's
    # request_queue.get() above.
    shutdown_event.wait()
finally:
    log.info("Stopping worker thread and MQTT client...")

    # Ask the worker to stop pulling any further queued requests once
    # this sentinel is seen...
    request_queue.put(SHUTDOWN_SENTINEL)

    # ...and force-close any Ollama connection currently in flight - whether
    # it's still waiting to connect, waiting for Ollama's first byte (the
    # 30-80s cold-start delay), or waiting between streamed chunks. This
    # unblocks the worker's blocked socket read almost immediately,
    # independent of --stream-idle-timeout or how long the generation
    # would otherwise have taken.
    #
    # socket.shutdown() rather than just close(): shutdown() tells the
    # kernel to abort any in-progress read/write on that socket right now,
    # which is what actually wakes up a recv() blocked on it in the worker
    # thread. close() alone just drops this thread's reference to the fd
    # and doesn't reliably interrupt another thread's blocking call on it -
    # worth doing both, shutdown() first, close() after.
    #
    # Note this only guarantees the *client side* stops waiting - whether
    # the Ollama server actually halts inference and frees GPU/VRAM on a
    # dropped connection is version- and model-dependent and not something
    # this script can guarantee; worth checking `ollama ps` / jtop after a
    # cancel on your actual setup if that matters to you, rather than
    # assuming it.
    with active_responses_lock:
        for conn in list(active_responses.values()):
            try:
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already closed/disconnected - nothing to unblock
            conn.close()

    worker_thread.join(timeout=SHUTDOWN_TIMEOUT)

    if worker_thread.is_alive():
        log.warning(
            f"Worker thread did not finish within {SHUTDOWN_TIMEOUT}s "
            "even after cancelling; proceeding with shutdown anyway - "
            "the thread is a daemon and will be discarded on process exit."
        )
    else:
        log.debug("Worker thread exited cleanly.")

    client.loop_stop()
    client.disconnect()
    log.info("Shutdown complete.")
