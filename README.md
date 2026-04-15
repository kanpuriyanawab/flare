# Flare

> Run open-source LLMs on your own cloud. OpenAI-compatible API, zero GPU cost when idle.

Flare deploys models from a curated registry (Llama, Qwen3, Gemma, Kimi, GGUF and more) onto your AWS/GCP/Azure/Kubernetes account and exposes a single OpenAI-compatible endpoint. Models scale to zero when idle and wake automatically on the next request — no GPU spend between calls.

Built on [SkyPilot](https://github.com/skypilot-org/skypilot). Apache 2.0.

---

## Who is this for?

- **ML platform engineers** — set up Flare once, give your team a stable endpoint. Skip this to [Platform Setup →](#platform-setup-ml-platform-engineer)
- **Application engineers** — call the endpoint like you would OpenAI. Skip this to [Using the API →](#using-the-api-application-engineer)

---

## Platform Setup (ML Platform Engineer)

You run this section once. Your team never needs to touch it.

### 1. Install

```bash
pip install flare-deploy[aws]        # AWS
pip install flare-deploy[gcp]        # GCP
pip install flare-deploy[azure]      # Azure
pip install flare-deploy[kubernetes] # Kubernetes
```

Requires Python 3.9+. SkyPilot is pulled in as a dependency of the cloud extra.

### 2. Validate credentials

```bash
flare init --infra aws               # checks aws sts get-caller-identity
flare init --infra gcp               # checks gcloud auth
flare init --infra kubernetes        # checks kubectl cluster-info
```

This writes `~/.flare/config.yaml` and installs SkyPilot if missing.

### 3. Choose your models — `models.yaml`

Create a `models.yaml` that declares what your team needs:

```yaml
infra: aws
region: us-east-1       # optional — SkyPilot auto-selects if omitted

defaults:
  idle_timeout: 15m     # scale to zero after 15 min of inactivity
  min_replicas: 0       # no GPU cost when idle
  max_replicas: 3       # autoscale up to 3 replicas under load

models:
  # Lightweight, fast — good for most tasks
  - name: qwen3-8b
    gpu: "L4:1"
    idle_timeout: 10m
    max_replicas: 5

  # High-quality reasoning — for complex tasks
  - name: qwen3-72b
    gpu: "A100:4"
    idle_timeout: 20m
    max_replicas: 2

  # Ultra-cheap GGUF (4-bit) — for high-volume, cost-sensitive workloads
  - name: llama-3.1-8b-q4km
    gpu: "T4:1"
    idle_timeout: 10m
    max_replicas: 10

  # Multimodal vision model
  - name: qwen2.5-vl-7b
    gpu: "L4:1"
    idle_timeout: 15m
```

Browse what's available:

```bash
flare catalog                        # all 32 models
flare catalog --family qwen          # filter by family
flare catalog --tag gguf             # GGUF quantized models (cheap)
flare catalog --capability vision    # multimodal models
flare catalog --capability reasoning # reasoning/thinking models
```

### 4. Deploy

```bash
flare apply models.yaml              # deploy everything in the file
```

Or deploy a single model ad-hoc:

```bash
flare deploy qwen3-8b                # uses registry defaults
flare deploy qwen3-72b --gpu A100:4 --max-replicas 2 --idle-timeout 20m
```

### 5. Start the gateway

The gateway is the single endpoint your team hits. It proxies requests to running models and queues requests for sleeping ones.

```bash
# Direct (development / single-node)
uvicorn flare.gateway.app:create_app --factory --host 0.0.0.0 --port 8080

# Docker Compose (recommended for production)
docker compose up -d

# Docker Compose with Redis (HA, multiple gateway replicas)
docker compose --profile ha up -d
```

Gateway environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLARE_GATEWAY_PORT` | `8080` | Listen port |
| `FLARE_DB_PATH` | `~/.flare/gateway.db` | SQLite state store path |
| `FLARE_REDIS_URL` | — | Redis URL; enables HA queue mode |
| `FLARE_POLL_INTERVAL` | `30` | SkyPilot poll interval in seconds |
| `FLARE_REQUIRE_API_KEY` | `false` | Set `true` to enforce API key auth |
| `HF_TOKEN` | — | HuggingFace token for gated models (Llama, Gemma) |

### 6. Verify

```bash
flare model                          # check deployment states and endpoints
curl http://localhost:8080/health    # {"status": "ok"}
curl http://localhost:8080/v1/models # list deployed models
```

### Day-2 operations

```bash
flare model                          # live status: state, replicas, cost/hr, endpoint
flare logs qwen3-8b                  # stream logs from the active replica
flare stop qwen3-72b                 # scale to zero immediately (keeps config)
flare rm qwen3-72b                   # full teardown (removes cloud resources)
flare apply models.yaml              # re-run anytime to sync desired vs actual state
```

### Batch jobs (cost-optimized bulk inference)

For large offline workloads, batch jobs run on spot instances (60–80% cheaper) and write results to object storage:

```bash
# Input: JSONL file, one request per line
# {"id": "r1", "prompt": "Summarize: ..."}

flare batch submit qwen3-8b \
  --input s3://my-bucket/prompts.jsonl \
  --output s3://my-bucket/results/

flare batch status                   # check job progress
```

### Cost visibility

```bash
flare cost                           # last 7 days
flare cost --period 30d              # last 30 days
flare cost --period 7d --model qwen3-72b
```

Output shows GPU-hours, estimated spend, and comparison against equivalent OpenAI API calls.

### Cost model

| Scenario | GPU cost |
|----------|----------|
| Model SLEEPING (min_replicas: 0) | **$0/hr** |
| Model RUNNING, 1× L4 | ~$0.54/hr |
| Model RUNNING, 1× A100 | ~$3.21/hr |
| Model RUNNING, 4× A100 | ~$12.84/hr |
| Batch job on spot (e.g. A100) | ~$1.00–1.50/hr (vs $3.21 on-demand) |

Set `idle_timeout` aggressively (10–15m) if your traffic is bursty. The gateway queues requests during the ~2–5 minute cold-start and delivers them once the model is live.

### Model registry

32 pre-validated configurations across 11 families. GPU specs, tensor parallelism, context lengths, and startup times are all pre-filled.

| Family | Models | Notes |
|--------|--------|-------|
| **Llama** (Meta) | 3.1-8B, 3.1-70B, 3.2-3B, 3.3-70B, 4-Scout-17B, 4-Maverick-17B | Llama 4 requires H100/H200 |
| **Qwen3** (Alibaba) | 8B, 14B, 32B, 72B | Built-in reasoning/thinking mode |
| **Qwen2.5-VL** | 7B, 72B | Vision + text |
| **Gemma 3** (Google) | 2B, 4B, 9B, 27B | Requires `HF_TOKEN` |
| **GLM** (THUDM) | GLM-4-9B, GLM-Z1-32B | Strong Chinese-English bilingual |
| **Kimi K2** (Moonshot) | Kimi-K2 | Requires H200:8 |
| **Mistral** | Mistral-7B-v0.3, Mixtral-8x7B | |
| **DeepSeek** | R1-Distill-8B, R1-Distill-32B | Strong math + coding |
| **Phi** (Microsoft) | Phi-3-Mini-4K, Phi-3-Medium-4K | |
| **GPT-OSS** (OpenAI) | gpt-oss-20b | Requires special vLLM build |
| **GGUF** | 6 models (Q4_K_M) | Llama-3, Llama-3.1, Qwen3, Gemma-3, Mistral-7B, DeepSeek-R1-7B |

To add a model: open a PR with a new YAML file in `registry/<family>/`. No code changes required.

---

## Using the API (Application Engineer)

Your ML platform team has set up Flare and given you a gateway URL (e.g. `http://flare.internal:8080`). Use it exactly like the OpenAI API.

### Drop-in replacement for OpenAI

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://flare.internal:8080/v1",
    api_key="none",          # or your team's API key if enforced
)

# Chat completions — identical to OpenAI
response = client.chat.completions.create(
    model="qwen3-8b",        # use flare catalog to see available models
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformer attention in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

```python
# Streaming
stream = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "Write a Python function to reverse a string."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

```python
# Text completions
response = client.completions.create(
    model="mistral-7b-v03",
    prompt="The capital of France is",
    max_tokens=10,
)
print(response.choices[0].text)
```

### Handling cold starts

When a model has been idle it goes to sleep (zero GPU cost). The first request after sleep returns **HTTP 202** instead of 200. The OpenAI SDK will raise an exception on 202 — handle it or use `FlareClient` which handles this automatically.

**Option A — FlareClient (recommended, auto-handles cold starts)**

```python
from flare.sdk import FlareClient

client = FlareClient(
    base_url="http://flare.internal:8080/v1",
    api_key="your-key",      # optional
    poll_interval=5.0,       # how often to check if model is ready (seconds)
    max_wait=600,            # give up after 10 minutes
)

# Blocks until response is ready — no 202 handling needed
response = client.chat(
    "qwen3-72b",
    [{"role": "user", "content": "Solve: integral of x^2 from 0 to 1"}],
)
print(response["choices"][0]["message"]["content"])

# Async version
response = await client.achat(
    "qwen3-8b",
    [{"role": "user", "content": "Hello!"}],
)
```

**Option B — raw OpenAI client with 202 handling**

```python
import httpx, time

def chat_with_retry(base_url: str, model: str, messages: list, api_key: str = "none") -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}

    resp = httpx.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers)

    if resp.status_code == 200:
        return resp.json()

    if resp.status_code == 202:
        data = resp.json()
        request_id = data["request_id"]
        print(f"Model is waking up (~{data['estimated_wait_seconds']}s). Polling...")

        while True:
            time.sleep(5)
            poll = httpx.get(f"{base_url}/v1/requests/{request_id}", headers=headers)
            result = poll.json()
            if result["status"] == "complete":
                return result   # same shape as a 200 response
            if result["status"] == "failed":
                raise RuntimeError(result["error"])

    resp.raise_for_status()
```

### @serve decorator

For scripts and services that own their deployment lifecycle:

```python
from flare.sdk import serve

@serve(
    model="llama-3.3-70b",
    gpu="A100:4",
    min_replicas=0,          # scale to zero when not in use
    max_replicas=2,
    idle_timeout="20m",
    gateway_url="http://flare.internal:8080/v1",
)
class Summarizer:
    """Auto-deploys llama-3.3-70b on first instantiation if not running."""
    pass

summarizer = Summarizer()

result = summarizer.chat([
    {"role": "user", "content": "Summarize this article: ..."},
])
print(result["choices"][0]["message"]["content"])
```

### Available models

```bash
# Install flare-deploy to browse locally
pip install flare-deploy
flare catalog

# Or hit the gateway directly
curl http://flare.internal:8080/v1/models
```

Common models and their use cases:

| Model | Best for | Min GPU |
|-------|----------|---------|
| `qwen3-8b` | General chat, fast responses | L4 (1×) |
| `qwen3-72b` | Complex reasoning, high quality | A100 (4×) |
| `qwen3-8b-q4km` | High volume, cost-sensitive | T4 (1×) |
| `llama-3.3-70b` | Instruction following, multilingual | A100 (4×) |
| `llama-3.1-8b-q4km` | Bulk processing (GGUF, very cheap) | T4 (1×) |
| `qwen2.5-vl-7b` | Image + text understanding | L4 (1×) |
| `deepseek-r1-distill-8b` | Math, coding, chain-of-thought | L4 (1×) |
| `mistral-7b-v03` | Function calling, structured output | L4 (1×) |
| `gemma-3-9b` | General use, Google quality | L4 (1×) |
| `glm-4-9b` | Chinese-English bilingual | L4 (1×) |

### API reference

The gateway implements the OpenAI REST API. All standard fields are supported and forwarded verbatim to the model.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Embeddings |
| `/v1/models` | GET | List deployed models and their state |
| `/v1/requests/{id}` | GET | Poll a cold-start queued request |
| `/health` | GET | Gateway health check |

**202 response shape** (cold start):
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "model": "qwen3-72b",
  "estimated_wait_seconds": 300,
  "poll_url": "/v1/requests/550e8400-e29b-41d4-a716-446655440000",
  "message": "Model 'qwen3-72b' is waking up. Poll /v1/requests/... for status."
}
```

**Poll response — pending:**
```json
{
  "request_id": "550e8400-...",
  "status": "pending",
  "model": "qwen3-72b",
  "estimated_wait_seconds": 180,
  "created_at": "2025-04-15T10:00:00Z"
}
```

**Poll response — complete:**

Returns the full original model response body (same as a 200 from `/v1/chat/completions`).

---

## Architecture overview

```
Your code / curl
      │
      ▼
Flare Gateway  ─────────────────────────────────────────────────────┐
  POST /v1/chat/completions                                         │
      │                                                             │
      ├── model RUNNING? ──yes──► proxy request ──► model endpoint │
      │                                                             │
      └── model SLEEPING? ──────► queue request (SQLite/Redis)     │
                                   return HTTP 202 + request_id    │
                                   trigger wake-up                 │
                                                                   │
  Background poller (every 30s)                                    │
      polls SkyPilot → detects RUNNING                             │
      replays queued requests → stores response                    │
                                                                   │
  GET /v1/requests/{id}                                            │
      pending → {"status": "pending"}                              │
      complete → original model response body                      │
                                                                   │
SkyPilot/SkyServe  (manages your cloud VMs) ──────────────────────┘
      │
GPU VM: vLLM or llama-cpp-python (OpenAI-compatible, port 8080)
```

---

## License

Apache 2.0