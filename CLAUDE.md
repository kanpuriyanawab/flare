# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses pyenv to manage Python. The `Desktop/.python-version` file activates the `transformers` virtualenv which has no pip. Always prefix Python commands with `PYENV_VERSION=3.12.4`:

```bash
PYENV_VERSION=3.12.4 python -m flare.cli.main --help
PYENV_VERSION=3.12.4 python -m pytest tests/
PYENV_VERSION=3.12.4 python -m ruff check flare/
PYENV_VERSION=3.12.4 python -m mypy flare/
```

## Install

```bash
PYENV_VERSION=3.12.4 python -m pip install -e ".[dev]"          # core + dev tools
PYENV_VERSION=3.12.4 python -m pip install -e ".[dev,aws]"      # + SkyPilot for AWS
PYENV_VERSION=3.12.4 python -m pip install -e ".[dev,redis]"    # + Redis queue
```

## Run commands

```bash
# CLI (installed as 'flare' entry point)
PYENV_VERSION=3.12.4 python -m flare.cli.main catalog
PYENV_VERSION=3.12.4 python -m flare.cli.main catalog --tag gguf
PYENV_VERSION=3.12.4 python -m flare.cli.main deploy qwen3-8b --gpu L4:1

# Gateway
PYENV_VERSION=3.12.4 python -m uvicorn flare.gateway.app:create_app --factory --port 8080

# Tests
PYENV_VERSION=3.12.4 python -m pytest tests/
PYENV_VERSION=3.12.4 python -m pytest tests/test_registry.py -v   # single file

# Lint / type check
PYENV_VERSION=3.12.4 python -m ruff check flare/
PYENV_VERSION=3.12.4 python -m ruff format flare/
PYENV_VERSION=3.12.4 python -m mypy flare/
```

## Architecture

Flare is a thin orchestration layer on top of SkyPilot/SkyServe. It does not run any GPU workloads itself — it generates SkyPilot task YAMLs and calls `sky.serve.*` / `sky.jobs.*`.

### Data flow

```
CLI / SDK
  └─► providers/skypilot.py        # generates task YAML via Jinja2, calls sky.serve.*
        └─► SkyPilot/SkyServe      # provisions cloud VMs, runs vLLM / llama-cpp

Gateway (FastAPI, port 8080)
  ├─ routes/chat.py                # POST /v1/chat/completions|/v1/completions
  │     ├─ RUNNING → httpx proxy to model endpoint
  │     └─ SLEEPING/OFF → enqueue + 202, trigger wake-up
  ├─ routes/requests_route.py      # GET /v1/requests/{id}  (poll cold-start)
  ├─ routes/models_route.py        # GET /v1/models
  ├─ poller.py                     # asyncio background task: polls SkyPilot every 30s,
  │                                #   detects RUNNING transition, replays queued requests
  └─ queue/{sqlite,redis}_queue.py # persistent request buffer
```

### Key modules

| Path | Purpose |
|------|---------|
| `flare/core/config.py` | All Pydantic models: `ModelSpec`, `FlareConfig` (models.yaml), `GatewayConfig`, `DeploymentEntry` |
| `flare/core/state.py` | `DeploymentState` enum + `skypilot_status_to_state()` mapping |
| `flare/registry/loader.py` | Singleton `Registry` — loads all `registry/**/*.yaml` files, validates with `ModelSpec` |
| `flare/providers/skypilot.py` | `SkyPilotProvider` — Jinja2 template rendering + `sky.serve.*` calls |
| `flare/gateway/app.py` | FastAPI app factory (`create_app()`), wires queue + poller via lifespan |
| `flare/gateway/poller.py` | `DeploymentPoller` — holds in-memory state cache, triggers wake-up, replays queued requests |
| `flare/sdk/client.py` | `FlareClient` — sync/async, auto-polls on 202 response |
| `flare/sdk/decorators.py` | `@serve` decorator — wraps a class with `.chat()` + auto-deploy |

### Registry YAML schema

Two variants (both validated by `ModelSpec`):

**HuggingFace model** — requires `hf_model_id`, engine defaults to `vllm`:
```yaml
name: qwen3-8b
hf_model_id: "Qwen/Qwen3-8B"
serving:
  engine: vllm
  tensor_parallel: 1
  context_length: 32768
gpus:
  recommended: ["L4:1"]
  minimum: ["T4:1"]
  fallback: []
memory_gb: 18
```

**GGUF checkpoint** — requires `gguf:` block, engine auto-set to `llama-cpp`:
```yaml
name: qwen3-8b-q4km
gguf:
  repo_id: "Qwen/Qwen3-8B-GGUF"
  filename: "Qwen3-8B-Q4_K_M.gguf"
  quantization: "Q4_K_M"
serving:
  engine: llama-cpp
  n_gpu_layers: -1
  context_length: 32768
gpus:
  recommended: ["L4:1"]
  minimum: ["T4:1"]
memory_gb: 6
```

### Cold-start 202 flow

1. Request arrives for a SLEEPING model
2. `chat.py` enqueues `QueuedRequest` in SQLite, calls `poller.trigger_wakeup()`
3. Returns `HTTP 202 {"request_id": "...", "poll_url": "/v1/requests/{id}"}`
4. Background `DeploymentPoller._poll_loop()` detects the model transitions to RUNNING
5. `_replay_queued()` re-POSTs each buffered request to the now-live endpoint
6. Client polls `/v1/requests/{id}` — gets `status: complete` + original response body

`FlareClient` handles this automatically; raw OpenAI clients receive the 202 and must poll.

### SkyPilot task YAML generation

`SkyPilotProvider._render_task_yaml()` selects one of three Jinja2 templates:
- `_VLLM_TASK_TEMPLATE` — standard HF models
- `_SGLANG_TASK_TEMPLATE` — SGLang engine
- `_LLAMACPP_TASK_TEMPLATE` — GGUF models (downloads via `huggingface_hub`, runs `llama_cpp.server`)

All templates embed a `service:` block (SkyServe) with `readiness_probe`, `min_replicas`, `max_replicas`, and `downscale_delay_s` (derived from `idle_timeout`).

### Adding a new model to the registry

1. Create `registry/<family>/<name>.yaml` following the schema above
2. Run `PYENV_VERSION=3.12.4 python -m flare.cli.main catalog` to confirm it loads
3. No code changes needed — the registry loader auto-discovers all YAML files

### Phase 2 features

- **Batch jobs**: `flare batch submit` → `SkyPilotProvider.submit_batch_job()` → `sky.jobs.launch()` with spot instances
- **Cost tracking**: `sqlite_queue.py` maintains a `cost_records` table; `flare cost` reads it
- **Pre-warm schedule**: `schedule:` field in `models.yaml` (cron string); poller will call `trigger_wakeup()` before peak hours
- **API keys**: `api_keys` table in SQLite; enforced by middleware in `gateway/app.py` when `FLARE_REQUIRE_API_KEY=true`

## Environment variables (gateway)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLARE_GATEWAY_HOST` | `0.0.0.0` | Bind address |
| `FLARE_GATEWAY_PORT` | `8080` | Bind port |
| `FLARE_DB_PATH` | `~/.flare/gateway.db` | SQLite database path |
| `FLARE_REDIS_URL` | unset | Redis URL for HA queue |
| `FLARE_POLL_INTERVAL` | `30` | SkyPilot poll interval (seconds) |
| `FLARE_REQUIRE_API_KEY` | `false` | Enable API key enforcement |
| `HF_TOKEN` | unset | HuggingFace token for gated models |
