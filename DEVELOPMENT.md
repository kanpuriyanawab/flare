# Flare — Development Guide

This document covers the full system architecture, implementation status, and roadmap for contributors and maintainers.

---

## System Architecture

Flare is an orchestration layer. It owns no GPU workloads itself — it generates configuration, calls SkyPilot APIs, and runs a stateful HTTP gateway. The three moving parts are:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Flare CLI / Python SDK                                             │
│  flare deploy qwen3-8b   |   FlareClient().chat(...)               │
└────────────────┬────────────────────────────────────────────────────┘
                 │  Python API calls
┌────────────────▼────────────────────────────────────────────────────┐
│  Flare Gateway  (FastAPI, single process, port 8080)                │
│                                                                     │
│  ┌────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │  Route layer   │   │  DeploymentPoller │   │  Request Queue  │  │
│  │  /v1/chat/...  │   │  (asyncio task)   │   │  SQLite/Redis   │  │
│  │  /v1/models    │   │  polls SkyPilot   │   │                 │  │
│  │  /v1/requests  │   │  every 30s        │   │                 │  │
│  └───────┬────────┘   └────────┬──────────┘   └────────┬────────┘  │
│          │  proxy              │ state cache            │ enqueue / │
│          │  (RUNNING)          │                        │ replay    │
└──────────┼─────────────────────┼────────────────────────┼───────────┘
           │                     │                        │
┌──────────▼─────────────────────▼────────────────────────▼───────────┐
│  SkyPilot / SkyServe                                                │
│  sky.serve.up() / sky.serve.status() / sky.jobs.launch()           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  provisions cloud VMs
┌──────────────────────────────▼──────────────────────────────────────┐
│  Cloud GPU VM  (AWS / GCP / Azure / Kubernetes)                     │
│  vLLM  OR  llama-cpp-python  (OpenAI-compatible API, port 8080)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Component responsibilities

**`flare/providers/skypilot.py` — `SkyPilotProvider`**
The only place that calls SkyPilot. Renders one of three Jinja2 templates (vLLM, SGLang, llama-cpp) into a SkyPilot task YAML, writes it to a temp file, calls `sky.Task.from_yaml()`, then `sky.serve.up()`. All other provider methods wrap `sky.serve.status()`, `sky.serve.update()`, `sky.serve.down()`, and `sky.jobs.*`.

**`flare/gateway/poller.py` — `DeploymentPoller`**
Runs as a background `asyncio.Task` inside the gateway process. Every `poll_interval_seconds` it calls `sky.serve.status()` (synchronous, run in thread pool implicitly via FastAPI's event loop). When a model transitions to `RUNNING`, it calls `_replay_queued()` which re-POSTs all buffered requests via `httpx.AsyncClient` to the now-live model endpoint.

**`flare/gateway/queue/` — request buffer**
`BaseQueue` abstract interface implemented by:
- `SQLiteQueue` — default, single-node, `aiosqlite`-backed, persists across gateway restarts
- `RedisQueue` — for multi-replica gateway deployments (`pip install 'flare-deploy[redis]'`)

The queue stores the full raw HTTP request (method, headers, body as bytes) so it can be replayed byte-for-byte.

**`flare/registry/loader.py` — `Registry`**
Singleton. On first access, walks `registry/**/*.yaml`, validates each file against `ModelSpec` (Pydantic v2), and builds an in-memory `dict[str, ModelSpec]`. Invalid files are skipped with a warning (non-strict mode). The registry is the only place that knows about GPU requirements, serving engine, startup time, and GGUF download coordinates.

**`flare/core/config.py` — Pydantic models**
Single source of truth for all schemas:
- `ModelSpec` — registry entry (what a model needs)
- `FlareConfig` — user's `models.yaml` (what they want deployed)
- `DeploymentEntry` — one model in `models.yaml`
- `GatewayConfig` — gateway runtime config (from env vars)
- `DeploymentInfo` — runtime state returned by the provider
- `BatchJobSpec` — Phase 2 batch job parameters

---

## Cold-Start 202 Flow (detailed)

This is the most complex original engineering in Flare. SkyPilot models scale to zero when idle; the gateway must queue incoming requests and replay them once the model is live.

```
Client                    Gateway                   SkyPilot / VM
  │                          │                           │
  │  POST /v1/chat/completions│                           │
  │  {"model": "qwen3-8b"}   │                           │
  ├─────────────────────────►│                           │
  │                          │  state = SLEEPING         │
  │                          │  enqueue(QueuedRequest)   │
  │                          │  trigger_wakeup()─────────►sky.serve.update(min_replicas=1)
  │  HTTP 202                │                           │
  │  {"request_id": "abc",   │                           │  VM provisioning...
  │   "poll_url": "/v1/req…",│                           │  (300s typical)
  │   "estimated_wait": 300} │                           │
  │◄─────────────────────────│                           │
  │                          │                           │
  │  GET /v1/requests/abc    │  status=pending           │
  ├─────────────────────────►│◄──────────────────────────│
  │  {"status":"pending"}    │                           │
  │◄─────────────────────────│                           │
  │           ...polling...  │   poller detects RUNNING  │
  │                          │◄──────────────────────────│ sky_status="READY"
  │                          │  replay_queued("qwen3-8b")│
  │                          │  POST /v1/chat/completions────────────────────►│
  │                          │◄──────────────────────────────────────────────│ 200 OK
  │                          │  store response in queue  │
  │  GET /v1/requests/abc    │                           │
  ├─────────────────────────►│                           │
  │  HTTP 200 (model response│                           │
  │  body verbatim)          │                           │
  │◄─────────────────────────│                           │
```

`FlareClient` (`flare/sdk/client.py`) automates the polling loop. Raw `openai.OpenAI` clients receive the 202 and must handle polling themselves using the `request_id` from the response body.

---

## SkyPilot Task YAML Generation

`SkyPilotProvider._render_task_yaml()` selects a template based on `ModelSpec.serving.engine`:

| Engine | Template | Setup | Run command |
|--------|----------|-------|-------------|
| `vllm` | `_VLLM_TASK_TEMPLATE` | `uv pip install vllm` | `python -m vllm.entrypoints.openai.api_server` |
| `sglang` | `_SGLANG_TASK_TEMPLATE` | `uv pip install sglang[all]` | `python -m sglang.launch_server` |
| `llama-cpp` | `_LLAMACPP_TASK_TEMPLATE` | `uv pip install llama-cpp-python[server]` + `hf_hub_download()` | `python -m llama_cpp.server` |

All three templates embed a `service:` block (SkyServe config):
- `readiness_probe.path: /v1/models` (OpenAI models endpoint)
- `readiness_probe.initial_delay_seconds` = `ModelSpec.startup_time_seconds`
- `replica_policy.min_replicas` = 0 (scale to zero by default)
- `replica_policy.downscale_delay_s` = parsed from `idle_timeout` in `models.yaml`

GPU spec conversion: `"A100:4"` → `{A100: 4}` (SkyPilot accelerators syntax).

---

## Deployment State Machine

```
         flare deploy
              │
           OFF ──────────────────► PROVISIONING
                                        │
                              VM ready, model loaded
                                        │
                                    RUNNING ──── idle_timeout ──► SLEEPING
                                        │                              │
                                   flare stop                   new request arrives
                                        │                       (gateway wake-up)
                                       OFF                      PROVISIONING

              FAILED ◄─── any state on error
              FAILED ──── flare deploy ──► PROVISIONING
```

`skypilot_status_to_state()` in `flare/core/state.py` maps raw SkyPilot service status strings (`"READY"`, `"NO_REPLICA"`, `"CONTROLLER_INIT"`, etc.) to `DeploymentState` enum values.

---

## Database Schema (SQLite, `~/.flare/gateway.db`)

Three tables maintained by `SQLiteQueue.initialize()`:

**`queued_requests`** — cold-start request buffer
```sql
request_id       TEXT PRIMARY KEY
model_name       TEXT
path             TEXT          -- e.g. /v1/chat/completions
method           TEXT          -- POST
headers          TEXT          -- JSON
body             BLOB          -- raw request bytes
status           TEXT          -- queued | waking | replaying | complete | failed
created_at       TEXT
completed_at     TEXT
response_status  INTEGER
response_headers TEXT          -- JSON
response_body    BLOB          -- raw response bytes stored verbatim
error            TEXT
estimated_wait   INTEGER       -- seconds
```

**`cost_records`** — Phase 2 cost tracking
```sql
id               INTEGER PRIMARY KEY
model_name       TEXT
deployment_id    TEXT
gpu_type         TEXT          -- e.g. A100
gpu_count        INTEGER
mode             TEXT          -- on-demand | batch
started_at       TEXT
stopped_at       TEXT
total_seconds    REAL
```

**`api_keys`** — Phase 2 API key management
```sql
key_hash         TEXT PRIMARY KEY  -- SHA-256 of the raw key
name             TEXT
created_at       TEXT
last_used_at     TEXT
is_active        INTEGER
```

---

## Implementation Status

### Phase 1 — CLI + Registry + On-Demand Serving ✅ Complete

| Component | Status | Notes |
|-----------|--------|-------|
| Model registry (32 models) | ✅ | `registry/**/*.yaml`, auto-discovered |
| `ModelSpec` Pydantic schema | ✅ | Validates HF + GGUF variants |
| `FlareConfig` / `models.yaml` schema | ✅ | Idle timeout parsing, defaults resolution |
| `flare init` | ✅ | Credential validation for AWS/GCP/Azure/K8s |
| `flare catalog` | ✅ | Rich table, filterable by family/tag/capability/engine |
| `flare deploy` | ✅ | Single model, GPU override, replica config |
| `flare apply` | ✅ | Desired-state reconciliation (deploy/update/remove) |
| `flare model` | ✅ | Rich table, state colors, cost/hr, endpoint |
| `flare stop` | ✅ | Scale to zero via `sky.serve.update(min_replicas=0)` |
| `flare rm` | ✅ | Full teardown via `sky.serve.down()` |
| `flare logs` | ✅ | Streaming via `sky.serve.tail_logs()` |
| SkyPilot provider | ✅ | vLLM + SGLang + llama-cpp templates |
| GGUF checkpoint serving | ✅ | `llama-cpp-python[server]`, `hf_hub_download` |
| Gateway FastAPI app | ✅ | Lifespan-managed, CORS, health endpoint |
| Cold-start 202 queue | ✅ | SQLite (default) + Redis (HA) |
| Background poller | ✅ | Asyncio task, 30s interval, auto-replay |
| `/v1/chat/completions` proxy | ✅ | Streaming SSE passthrough |
| `/v1/models` endpoint | ✅ | OpenAI format |
| `/v1/requests/{id}` poll | ✅ | Returns raw response body on complete |
| `FlareClient` Python SDK | ✅ | Sync + async, auto-polls 202 |
| `@serve` decorator | ✅ | Auto-deploy on instantiation |
| `docker-compose.yml` | ✅ | Gateway + optional Redis profile |

### Phase 2 — Batch + Cost Visibility ✅ Implemented (needs integration testing)

| Component | Status | Notes |
|-----------|--------|-------|
| `flare batch submit` | ✅ | `sky.jobs.launch()` with spot instances |
| `flare batch status` | ✅ | `sky.jobs.queue()` wrapper |
| `flare cost --period` | ✅ | Reads `cost_records` from SQLite |
| Cost record accumulation | ⚠️ | Poller must call `record_deployment_start/stop` on transitions — not yet wired |
| Pre-warm `schedule:` field | ⚠️ | Schema supports it; poller cron execution not yet wired |
| API key enforcement | ✅ | Middleware in `gateway/app.py`, `FLARE_REQUIRE_API_KEY=true` |
| API key management CLI | ❌ | No `flare key add/list/revoke` commands yet |

### Phase 3 — Custom Models + Kubernetes (Not started)

| Feature | Notes |
|---------|-------|
| `@serve` for custom (non-registry) models | Schema extension needed |
| `flare catalog add <hf_id>` | Auto-generate YAML from HF model card heuristics |
| Helm chart for K8s gateway deployment | `charts/` directory |
| Team namespacing (per-team API keys) | Multi-tenant `api_keys` schema |
| Image build API | Docker SDK wrapper for custom container builds |

---

## Known Gaps & TODOs

### Must fix before production use

1. **Cost tracking not wired** — `record_deployment_start()` and `record_deployment_stop()` exist in `sqlite_queue.py` but `DeploymentPoller` does not call them on state transitions. `flare cost` will return empty results until this is connected.

2. **Pre-warm schedule not executed** — `DeploymentEntry.schedule` is parsed and stored but `DeploymentPoller` does not read it. Needs `apscheduler` or `croniter` integration in the poll loop.

3. **`sky.serve.update()` API** — The `scale_up()` and `stop()` provider methods call `sky.serve.update(min_replicas=N)`. Verify this is the correct SkyPilot API for the installed version; it may be `sky.serve.update(service_name, task, ...)`.

4. **No tests** — `tests/` directory is empty. Priority test areas: registry loading, Pydantic validation, gateway 202 flow (mock SkyPilot), `FlareClient` polling logic.

### Nice to have

5. **`flare key` commands** — `add`, `list`, `revoke` for API key management (Phase 2 table exists, no CLI surface).

6. **Gateway streaming replay** — Currently replayed responses are buffered in full before storing. Large streaming responses may exhaust memory. Should stream directly to the stored response.

7. **Multiple replicas log selection** — `flare logs` defaults to `replica=0`. Add `--replica all` to fan out.

8. **`flare apply` update path** — Currently skips models that already exist (`to_update`). Should diff GPU/replica config and call `sky.serve.update()` if changed.

9. **Registry `disk_gb` minimum** — Set to 1 to accommodate GGUF files. SkyPilot's minimum disk size may be higher; add a check in task YAML generation.

---

## Adding to the Registry

To add a model, create `registry/<family>/<name>.yaml`. No code changes required.

**HuggingFace model:**
```yaml
name: my-model-7b
display_name: "My Model 7B Instruct"
family: mymodel
version: "1"
description: "One-line description."
hf_model_id: "org/My-Model-7B-Instruct"
serving:
  engine: vllm          # vllm | sglang
  tensor_parallel: 1
  context_length: 32768
  max_model_len: 32768
  dtype: auto
  gpu_memory_utilization: 0.90
  extra_args: []        # verbatim flags appended to the serve command
gpus:
  recommended: ["L4:1", "A10G:1"]
  minimum: ["T4:1"]
  fallback: []
memory_gb: 18
disk_gb: 40
startup_time_seconds: 120
requires_hf_token: false
capabilities: [chat, text-generation]
tags: [mymodel, fast]
```

**GGUF checkpoint** (engine auto-set to `llama-cpp`):
```yaml
name: my-model-7b-q4km
display_name: "My Model 7B Q4_K_M (GGUF)"
family: mymodel
version: "1"
gguf:
  repo_id: "bartowski/My-Model-7B-GGUF"
  filename: "My-Model-7B-Q4_K_M.gguf"
  quantization: "Q4_K_M"
serving:
  engine: llama-cpp
  n_gpu_layers: -1
  context_length: 8192
  n_threads: 8
gpus:
  recommended: ["T4:1", "L4:1"]
  minimum: ["T4:1"]
  fallback: []
memory_gb: 5
disk_gb: 5
startup_time_seconds: 60
requires_hf_token: false
capabilities: [chat, text-generation]
tags: [mymodel, gguf, quantized]
```

Validate with:
```bash
PYENV_VERSION=3.12.4 python -m flare.cli.main catalog --family mymodel
```

---

## Dependency Map

```
flare/core/config.py       ← pydantic v2
flare/core/state.py        ← no deps
flare/core/exceptions.py   ← no deps
flare/core/constants.py    ← pathlib only

flare/registry/loader.py   ← core/config, core/exceptions, pyyaml

flare/providers/base.py    ← core/config, core/state
flare/providers/skypilot.py← providers/base, core/config, core/state, jinja2, sky (optional)

flare/gateway/queue/base.py      ← pydantic
flare/gateway/queue/sqlite_queue ← queue/base, aiosqlite
flare/gateway/queue/redis_queue  ← queue/base, redis (optional)
flare/gateway/poller.py          ← gateway/queue/base, core/state, httpx
flare/gateway/routes/chat.py     ← gateway/queue/base, core/state, registry/loader, httpx
flare/gateway/app.py             ← all gateway modules, fastapi

flare/sdk/client.py        ← httpx
flare/sdk/decorators.py    ← sdk/client, registry/loader, providers/skypilot, core/config

flare/cli/main.py          ← all cli/commands/*
flare/cli/commands/*       ← registry/loader, providers/skypilot, core/config, rich, click
```

`sky` (SkyPilot) is always an optional import — `SkyPilotProvider._check_sky()` catches `ImportError` and surfaces a clear install message. The gateway and registry work without SkyPilot installed.

---

## Roadmap

### v0.1 — Stabilization (current)
- [ ] Wire cost tracking into the poller state transitions
- [ ] Wire pre-warm `schedule:` cron execution in poller
- [ ] Fix `flare apply` update path (diff + `sky.serve.update`)
- [ ] Add `flare key add/list/revoke` commands
- [ ] Write test suite (registry, gateway 202 flow, SDK polling)
- [ ] Validate `sky.serve.update()` API against installed SkyPilot version

### v0.2 — Observability & UX
- [ ] `flare model --watch` — live-updating status table (Rich Live)
- [ ] Webhook callback on model RUNNING (alternative to polling)
- [ ] Per-request latency tracking in `cost_records`
- [ ] `flare logs --replica all` fan-out
- [ ] `flare catalog add <hf_id>` — auto-generate YAML from HF model card

### v0.3 — Custom Models & Kubernetes
- [ ] `@serve` for arbitrary non-registry models (custom HF IDs)
- [ ] Helm chart for gateway on K8s
- [ ] Team namespacing (per-team API keys, model access control)
- [ ] `flare image build` — Docker SDK wrapper for custom serving containers

### v0.4 — Enterprise
- [ ] Multi-gateway federation (route by team/project)
- [ ] SLO dashboards (Grafana integration)
- [ ] SSO / OIDC for API key issuance
- [ ] Audit log for all model requests
