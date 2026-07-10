# orin

MQTT-based services for an NVIDIA Jetson Orin Nano: a bridge to a local Ollama LLM server, and a system telemetry publisher.

**Contents:** [mqtt_llm.py](#mqtt_llmpy) · [mqtt_telemetry.py](#mqtt_telemetrypy) · [Running checks locally](#running-checks-locally) · [Reference](#reference) · [Design rationale](#design-rationale)

## mqtt_llm.py

Bridges MQTT to a local [Ollama](https://ollama.com) server: subscribes to a request topic, forwards each request to Ollama's `/api/generate`, streams the generated response back as a single MQTT reply, and publishes an online/offline presence status.

```bash
python3 mqtt_llm.py --broker 192.168.1.10 --topic orin/ollama
```

Requires an MQTT broker reachable at `--broker` and an Ollama server reachable at `--ollama-host`/`--ollama-port` (default `localhost:11434`). Full CLI, MQTT topics, and payload formats: see [Reference](#reference).

## mqtt_telemetry.py

Publishes Jetson system telemetry (CPU/GPU load, RAM, temperatures, fan, power) to MQTT on a fixed interval, via [jetson-stats](https://github.com/rbonghi/jetson_stats). Only runs on actual Jetson hardware with the `jetson_stats` service active.

```bash
python3 mqtt_telemetry.py --broker 192.168.1.10 --topic orin/status --interval 10
```

## Running checks locally

Requires [uv](https://docs.astral.sh/uv/). `uv sync` sets up everything else, including a matching Python - no Jetson hardware needed, the test suite fakes out the Jetson-only parts.

```bash
./scripts/ci.sh          # everything CI runs: lint + type check + tests
./scripts/lint.sh        # ruff check
./scripts/typecheck.sh   # mypy
./scripts/test.sh        # pytest
```

The integration tests spin up a real local Mosquitto broker as a subprocess, so the `mosquitto` binary needs to be on `PATH` (e.g. `apt install mosquitto`); `scripts/test.sh` warns if it's missing.

## Reference

<details>
<summary><strong>mqtt_llm.py CLI options</strong></summary>

| Flag | Default | Description |
|---|---|---|
| `--broker` (required) | – | MQTT broker IP or hostname |
| `--port` | `1883` | MQTT broker port |
| `--username` | – | MQTT username |
| `--credpath` | – | Path to a file holding the MQTT password (e.g. systemd `LoadCredential=`) |
| `--topic` | `orin/ollama` | Base MQTT topic; see topics below |
| `--ollama-host` | `localhost` | Ollama server host |
| `--ollama-port` | `11434` | Ollama server port |
| `--connect-timeout` | `5.0` | Max seconds to establish the Ollama TCP connection |
| `--stream-timeout` | `120.0` | Max seconds of silence between streamed chunks before giving up |
| `--shutdown-timeout` | `5.0` | Max seconds to wait for in-flight work to finish on shutdown |
| `--interval` | `10` | Status message publish interval in seconds |
| `--loglevel` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |

</details>

<details>
<summary><strong>mqtt_llm.py MQTT topics and payloads</strong></summary>

- **`{topic}/request`** (subscribed, qos 1) - a request:
  ```json
  {"request_id": "abc123", "model": "llama3", "system": "You are helpful.", "user": "Say hi",
   "temperature": 0.7, "top_p": null, "top_k": null}
  ```
  `request_id`, `model`, `system`, `user` are required; `temperature`/`top_p`/`top_k` are optional.

- **`{topic}/response`** (published, qos 1) - the reply:
  ```json
  {"request_id": "abc123", "message": "Hi there!", "error_code": 0, "error_message": "OK"}
  ```
  `error_code`: `0` OK, `1` invalid JSON, `2` schema validation error, `3` Ollama error.

- **`{topic}/status`** (published, qos 1, retained) - presence plus the current request queue:
  ```json
  {
    "status": "online", "heartbeat": 1735689600,
    "queue": [
      {"request_id": "abc123", "response_chars": -1, "thinking_chars": -1},
      {"request_id": "def456", "response_chars": 0, "thinking_chars": -1},
      {"request_id": "ghi789", "response_chars": 42, "thinking_chars": 17}
    ]
  }
  ```
  `queue` lists every request not yet fully processed, in the order it will be (or is being) processed. Each entry's `response_chars`/`thinking_chars` track that request's own generation progress independently (a "thinking" phase and a final-response phase can advance separately for reasoning models). `response_chars` is `-1` queued but not yet started, `0` started but no response text generated yet, otherwise the response character count so far. `thinking_chars` follows the same `-1`/`0`/count shape, but since not every model streams a "thinking" field at all, `-1` also covers "no thinking activity observed for this request yet" - it only leaves `-1` once Ollama actually sends one (even an empty one, which reads as `0`), and simply stays `-1` for the whole request if the model never does. `heartbeat` is `int(time.time())` at publish time, so a subscriber can tell a retained message apart from a live one.

  Published on the same schedule as `mqtt_telemetry.py`'s telemetry (every `--interval` seconds), plus immediately whenever a request is enqueued or finishes (successfully, with an error, or cancelled by shutdown) - so a requestor can see their `request_id` land in the queue right away, or notice it disappear again without a normal `{topic}/response` reply (e.g. an error that pulled it back out before it ever ran).

  When offline (either the MQTT Last Will firing on an ungraceful exit, or the explicit publish during a clean shutdown), `heartbeat` and `queue` are `null` - `status` is the only field guaranteed meaningful at that point. A clean shutdown also publishes this offline payload explicitly, since a graceful disconnect cancels LWT delivery.

</details>

<details>
<summary><strong>mqtt_telemetry.py CLI options</strong></summary>

| Flag | Default | Description |
|---|---|---|
| `--broker` (required) | – | MQTT broker IP or hostname |
| `--port` | `1883` | MQTT broker port |
| `--username` | – | MQTT username |
| `--credpath` | – | Path to a file holding the MQTT password (e.g. systemd `LoadCredential=`) |
| `--topic` | `orin/status` | MQTT topic telemetry is published to |
| `--interval` | `10` | Publish interval in seconds |
| `--loglevel` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |

</details>

<details>
<summary><strong>mqtt_telemetry.py MQTT topic and payload</strong></summary>

Published to `{topic}` (qos 1) on every interval, and once more (qos 1, retained) with `heartbeat: 0, status: "offline"` on shutdown - also the MQTT Last Will, for the same ungraceful-exit reason as `mqtt_llm.py`.

```json
{
  "heartbeat": 1735689600, "status": "online",
  "uptime_s": 3725, "cpu_avg": 5.5, "cpu_max": 8.0,
  "ram_used_ratio": 0.35, "swap_used_ratio": 0.0, "gpu_load": 12.5,
  "fan_pwm": 60, "fan_rpm": 1500,
  "temp_max": 45.0, "temp_cpu": 45.0, "temp_gpu": 44.0,
  "power_total": 4.2
}
```

Any field is `null` if the corresponding sensor/stat isn't available. If jtop itself is unavailable (e.g. the `jetson_stats` service restarting for an update), that's reported the same way, live on `{topic}` with `status: "offline"` and every sensor field `null` - but with a real, current `heartbeat`, distinguishing "process alive, sensors degraded" from the `heartbeat: 0` shutdown/LWT message. The service retries in the background on its own (starting at 5s, backing off to a 60s cap the longer the outage lasts, resetting once jtop is reachable again) and flips back to `status: "online"` automatically - no restart needed.

</details>

<details>
<summary><strong>Repository layout</strong></summary>

- `mqtt_llm.py`, `mqtt_telemetry.py` - the two services, each a single standalone script
- `tests/` - pytest suite: a real local Mosquitto broker, a fake Ollama HTTP server, and a faked `jtop` module (see `tests/conftest.py`)
- `scripts/` - local entry points for the same checks CI runs
- `.github/workflows/ci.yml` - CI: `mypy`, `ruff check`, `pytest`, each also runnable via `scripts/`
- `pyproject.toml` - single source of truth for dependencies (`uv`) and tool config (`ruff`, `mypy`, `pytest`)

</details>

## Design rationale

The "why" behind non-obvious code, kept out of the way of the top-level flow in both scripts - each links back here from a short pointer comment. See also `CLAUDE.md`, which carries the same knowledge organized around what future edits need to preserve.

### mqtt_llm.py

<details>
<summary><strong>Connection and shutdown races</strong></summary>
<a id="connect-shutdown-races"></a>
<a id="activegeneration-socket-capture"></a>

`client.connect()` only sends the CONNECT packet - the CONNACK that flips `is_connected()` to `True` arrives asynchronously on the network loop thread. `wait_for_mqtt_connection()` blocks on an `Event` until that CONNACK lands (or shutdown is requested) before anything assumes a live connection. Without it, a shutdown signal arriving between `connect()` and the CONNACK could see `is_connected()` return `False` (skipping the explicit offline publish), only for the CONNACK to land moments later and let the eventual `disconnect()` go out "clean" - which cancels Last Will delivery too. Net effect: no offline announcement at all. `handle_exit()` also sets the same event so a shutdown request wakes this wait immediately rather than waiting out the log interval.

The shutdown sequence itself, once triggered, runs once and in order: drain the request queue, interrupt any in-flight Ollama calls (see below), join the worker thread with a timeout, publish the offline status explicitly (a graceful `disconnect()` cancels the Last Will, so a clean exit needs its own offline publish), then stop the network loop and disconnect. Both the worker and status-publisher threads are daemons, so a stuck thread past its join timeout only delays shutdown, never blocks it forever - the OS reclaims it on process exit.

`ActiveGeneration` exists to make that "interrupt in-flight calls" step possible. `http.client`'s `getresponse()` closes and nulls `HTTPConnection.sock` as soon as it reads a response with no `Content-Length` (true for our streamed NDJSON responses) - effectively handing the live socket to the response object instead of leaving it on the connection. So `conn.sock` can't be relied on once streaming has started. `ActiveGeneration` captures the raw socket reference right after `connect()`, before `getresponse()` can null it, and calls `shutdown(SOCK_RDWR)` on it directly during teardown - the underlying socket object stays valid even though `conn`'s reference to it is gone, and `shutdown()` still interrupts a thread blocked reading it.

Registering a call in `active_responses` and checking `shutdown_event` happen under the same `active_responses_lock` acquisition in `stream_ollama_generate`. That pairing closes a register-vs-shutdown race: without it, a request could be dequeued just as shutdown begins, register itself a moment after the shutdown block already took its snapshot of `active_responses`, and then run uninterrupted to completion.

</details>

<details>
<summary><strong>request_id is not a key</strong></summary>
<a id="request-id-is-not-a-key"></a>

`request_id` is caller-supplied, and pydantic only validates it as `str` - not for uniqueness or non-emptiness. Two, or two hundred, queued requests can legitimately share one `request_id`, or all use `""`. Nothing internal may use it as a lookup key.

`queue_status` is keyed instead by `_queue_seq_counter`, a private, always-unique counter assigned per request at enqueue time (`seq` in `on_message`/`stream_ollama_generate`/`ollama_worker`). Each entry's own `request_id` field is carried purely to report back to subscribers, never as an identity. This was a real, reported bug before the fix: duplicate-`request_id` requests used to collapse onto one `queue_status` entry.

Insertion order into `queue_status` also doubles as the queue's processing order (it's a plain dict, and `request_queue` is FIFO), so no separate structure is needed to answer "what order will these process in".

</details>

<details>
<summary><strong>response_chars / thinking_chars semantics</strong></summary>
<a id="response-thinking-chars-semantics"></a>

Each queue entry tracks its own `response_chars` and `thinking_chars`, because a "thinking" phase and a final-response phase can advance independently for reasoning models. Both follow the same shape: `-1` queued but not started, `0` started but nothing generated yet, otherwise the running character count.

`thinking_chars`'s `-1` carries a second meaning beyond "queued": "no thinking activity observed yet". Not every model streams a `thinking` field at all, so it only leaves `-1` once `stream_ollama_generate` actually sees a `"thinking"` key in a chunk (even an empty one, which then reads as `0`) - for a model that never sends one, it simply stays `-1` for the whole request. This was also a real, reported bug: it used to default to `0` alongside `response_chars` the moment a request started, even for models that never stream a `thinking` field.

`publish_status()` wraps snapshotting `queue_status` and publishing as one atomic unit under `status_publish_lock`, across every thread that calls it (`on_message`, `ollama_worker`, `status_publisher`). Without it, two concurrent publishes could interleave so a stale, smaller snapshot reaches the broker *after* a fresher, larger one and overwrites it.

</details>

<details>
<summary><strong>Ollama stream parsing robustness</strong></summary>
<a id="ollama-stream-parsing"></a>

Two defensive checks in `stream_ollama_generate` guard against a misbehaving Ollama server rather than a well-formed one: a chunk that decodes to JSON but isn't an object is rejected explicitly (it would otherwise fail confusingly inside `chunk.get(...)`), and a connection that ends - closes, resets, hits EOF - without a final `"done"` chunk is treated as an error, not a complete response. Without the second check, a caller could easily mistake a truncated stream for a finished one.

</details>

### mqtt_telemetry.py

<details>
<summary><strong>Connection race (telemetry service)</strong></summary>
<a id="telemetry-connect-race"></a>

Same underlying issue as `mqtt_llm.py`'s connection race (see above): `client.connect()` only sends the CONNECT packet, and the CONNACK that flips `is_connected()` to `True` lands asynchronously. `wait_for_mqtt_connection()` blocks on it before `jtop()` is ever started, so a shutdown signal arriving early can't slip through the same "looks connected but the Last Will already fired" gap. paho's own `reconnect_delay_set` backoff is retrying in the background regardless; this just waits for it to succeed (or a shutdown signal to interrupt the wait), logging periodically so a persistent outage stays visible.

</details>

<details>
<summary><strong>jtop lifecycle: reads, staleness, and slow closes</strong></summary>
<a id="jtop-lifecycle"></a>

`jetson.stats`/`.fan` are plain in-memory reads with zero I/O - they cannot hang, but they also never raise for a lost connection. jtop's background reader thread catches that internally and only surfaces it if the caller calls `jetson.ok()`. `publish_telemetry()` calls `jetson.ok(spin=True)` specifically for this; without it, a mid-run outage would silently return stale data forever, with no error and no retry. `spin=True` skips jtop's normal blocking wait for a fresh sample, since the publish loop already paces itself via `--interval`.

`jtop.__exit__` does not stop the background thread or close the connection - `close()` does, but `close()` joins an internal thread that shells out to `dpkg -l`/`nvcc`/`opencv_version` with no timeout of its own, and `dpkg` can be locked for the length of a whole system update - exactly the scenario the retry loop exists to survive. So `close()` is never called synchronously; `_close_jtop_in_background()` runs it in its own daemon thread, best-effort, so a stuck close only delays that one instance's cleanup, never recovery or shutdown. Don't "simplify" this back to an inline `jetsonTop.close()` call.

`jtop()` failing to open and jtop breaking mid-run are handled identically in the main loop: discard whatever jtop state exists and retry a fresh `jtop()` instance after a backoff delay (`JTOP_RETRY_DELAY_MIN` doubling to `JTOP_RETRY_DELAY_MAX`, reset once an open succeeds) - a fresh instance isn't free (see above), so backing off avoids piling up retries for the length of an outage. `SystemExit`, raised by `handle_exit()` on SIGTERM/SIGINT, is not an `Exception` subclass, so it always passes through this retry loop untouched, including while blocked in the `time.sleep()` between attempts.

A jtop outage is reported on the telemetry topic itself - real, current `heartbeat`, every sensor field `null`, `status: "offline"` - distinct from the `heartbeat: 0` message that's the MQTT Last Will and the explicit clean-shutdown publish. The former means "process alive, sensors degraded, watch it recover"; the latter means "the process itself is gone."

</details>
