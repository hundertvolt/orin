# orin

MQTT-based services for an NVIDIA Jetson Orin Nano: a bridge to a local Ollama LLM server, and a system telemetry publisher.

**Contents:** [mqtt_llm.py](#mqtt_llmpy) · [mqtt_telemetry.py](#mqtt_telemetrypy) · [Running checks locally](#running-checks-locally) · [Reference](#reference)

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

- **`{topic}/status`** (published, qos 1, retained) - presence: `{"status": "online"}` / `{"status": "offline"}`. Set as the MQTT Last Will so an ungraceful exit (crash, power loss, `kill -9`) is still announced; a clean shutdown also publishes the offline status explicitly, since a graceful disconnect cancels LWT delivery.

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

Any field is `null` if the corresponding sensor/stat isn't available.

</details>

<details>
<summary><strong>Repository layout</strong></summary>

- `mqtt_llm.py`, `mqtt_telemetry.py` - the two services, each a single standalone script
- `tests/` - pytest suite: a real local Mosquitto broker, a fake Ollama HTTP server, and a faked `jtop` module (see `tests/conftest.py`)
- `scripts/` - local entry points for the same checks CI runs
- `.github/workflows/ci.yml` - CI: `mypy`, `ruff check`, `pytest`, each also runnable via `scripts/`
- `pyproject.toml` - single source of truth for dependencies (`uv`) and tool config (`ruff`, `mypy`, `pytest`)

</details>
