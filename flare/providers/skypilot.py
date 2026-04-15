"""SkyPilot provider: wraps sky.serve.* and sky.jobs.* APIs."""

from __future__ import annotations

import logging
import textwrap
from typing import Iterator

from jinja2 import Environment, BaseLoader

from flare.core.config import (
    BatchJobSpec,
    DeploymentEntry,
    DeploymentInfo,
    GlobalDefaults,
    ModelSpec,
    ServingEngine,
)
from flare.core.state import DeploymentState, skypilot_status_to_state
from flare.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jinja2 templates for SkyPilot task YAML
# ---------------------------------------------------------------------------

_VLLM_TASK_TEMPLATE = """\
name: {{ name }}

resources:
  accelerators: {{ gpu_spec }}
  disk_size: {{ disk_gb }}
  ports:
    - {{ port }}

envs:
  MODEL_NAME: "{{ hf_model_id }}"
  HF_TOKEN: "${HF_TOKEN:-}"

setup: |
  pip install uv -q
  uv pip install --system "vllm>=0.7.0" huggingface_hub -q
  {% if requires_hf_token %}
  huggingface-cli login --token $HF_TOKEN
  {% endif %}

run: |
  python -m vllm.entrypoints.openai.api_server \\
    --model $MODEL_NAME \\
    --host 0.0.0.0 \\
    --port {{ port }} \\
    --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \\
    --dtype {{ dtype }} \\
    --max-model-len {{ max_model_len }} \\
    --gpu-memory-utilization {{ gpu_memory_utilization }} \\
    {% for arg in extra_args %}
    {{ arg }} \\
    {% endfor %}
    --trust-remote-code

service:
  readiness_probe:
    path: /v1/models
    initial_delay_seconds: {{ startup_time_seconds }}
  replica_policy:
    min_replicas: {{ min_replicas }}
    max_replicas: {{ max_replicas }}
    target_qps_per_replica: 5
    upscale_delay_s: 60
    downscale_delay_s: {{ idle_timeout_seconds }}
"""

_SGLANG_TASK_TEMPLATE = """\
name: {{ name }}

resources:
  accelerators: {{ gpu_spec }}
  disk_size: {{ disk_gb }}
  ports:
    - {{ port }}

envs:
  MODEL_NAME: "{{ hf_model_id }}"
  HF_TOKEN: "${HF_TOKEN:-}"

setup: |
  pip install uv -q
  uv pip install --system "sglang[all]>=0.4.0" huggingface_hub -q
  {% if requires_hf_token %}
  huggingface-cli login --token $HF_TOKEN
  {% endif %}

run: |
  python -m sglang.launch_server \\
    --model-path $MODEL_NAME \\
    --host 0.0.0.0 \\
    --port {{ port }} \\
    --tp {{ tensor_parallel }} \\
    --dtype {{ dtype }} \\
    {% for arg in extra_args %}
    {{ arg }} \\
    {% endfor %}
    --trust-remote-code

service:
  readiness_probe:
    path: /v1/models
    initial_delay_seconds: {{ startup_time_seconds }}
  replica_policy:
    min_replicas: {{ min_replicas }}
    max_replicas: {{ max_replicas }}
    target_qps_per_replica: 5
    upscale_delay_s: 60
    downscale_delay_s: {{ idle_timeout_seconds }}
"""

_LLAMACPP_TASK_TEMPLATE = """\
name: {{ name }}

resources:
  accelerators: {{ gpu_spec }}
  disk_size: {{ disk_gb }}
  ports:
    - {{ port }}

envs:
  GGUF_REPO: "{{ gguf_repo_id }}"
  GGUF_FILENAME: "{{ gguf_filename }}"

setup: |
  pip install uv -q
  uv pip install --system "llama-cpp-python[server]" huggingface_hub -q
  # Download the GGUF checkpoint
  python -c "
  from huggingface_hub import hf_hub_download
  import os
  path = hf_hub_download(repo_id=os.environ['GGUF_REPO'], filename=os.environ['GGUF_FILENAME'])
  print(f'Downloaded to: {path}')
  # Symlink to a fixed path
  import shutil
  shutil.copy2(path, '/tmp/model.gguf')
  print('Copied to /tmp/model.gguf')
  "

run: |
  python -m llama_cpp.server \\
    --model /tmp/model.gguf \\
    --host 0.0.0.0 \\
    --port {{ port }} \\
    --n_gpu_layers {{ n_gpu_layers }} \\
    --n_ctx {{ context_length }} \\
    --n_threads {{ n_threads }}

service:
  readiness_probe:
    path: /v1/models
    initial_delay_seconds: {{ startup_time_seconds }}
  replica_policy:
    min_replicas: {{ min_replicas }}
    max_replicas: {{ max_replicas }}
    target_qps_per_replica: 10
    upscale_delay_s: 30
    downscale_delay_s: {{ idle_timeout_seconds }}
"""

_BATCH_TASK_TEMPLATE = """\
name: {{ name }}-batch

resources:
  accelerators: {{ gpu_spec }}
  disk_size: {{ disk_gb }}
  use_spot: {{ use_spot }}

envs:
  MODEL_NAME: "{{ hf_model_id }}"
  INPUT_PATH: "{{ input_path }}"
  OUTPUT_PATH: "{{ output_path }}"
  HF_TOKEN: "${HF_TOKEN:-}"

setup: |
  pip install uv -q
  uv pip install --system "vllm>=0.7.0" huggingface_hub -q
  {% if requires_hf_token %}
  huggingface-cli login --token $HF_TOKEN
  {% endif %}

run: |
  python -c "
  import json, os
  from vllm import LLM, SamplingParams

  llm = LLM(model=os.environ['MODEL_NAME'])
  params = SamplingParams(temperature=0.7, max_tokens=2048)

  with open(os.environ['INPUT_PATH']) as f:
      lines = [json.loads(l) for l in f if l.strip()]

  prompts = [l.get('prompt', l.get('messages', '')) for l in lines]
  outputs = llm.generate(prompts, params)

  with open('/tmp/batch_results.jsonl', 'w') as out:
      for line, output in zip(lines, outputs):
          result = {'id': line.get('id'), 'response': output.outputs[0].text}
          out.write(json.dumps(result) + '\n')

  # Upload results
  output_path = os.environ['OUTPUT_PATH']
  if output_path.startswith('s3://'):
      import subprocess
      subprocess.run(['aws', 's3', 'cp', '/tmp/batch_results.jsonl', output_path], check=True)
  elif output_path.startswith('gs://'):
      import subprocess
      subprocess.run(['gsutil', 'cp', '/tmp/batch_results.jsonl', output_path], check=True)
  else:
      import shutil
      shutil.copy('/tmp/batch_results.jsonl', output_path)

  print(f'Results written to: {output_path}')
  "
"""


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

_JINJA_ENV = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)


def _parse_idle_timeout_to_seconds(timeout_str: str) -> int:
    """Convert '15m' → 900."""
    from flare.core.config import _parse_idle_timeout
    return _parse_idle_timeout(timeout_str)


def _gpu_spec_to_sky_accelerators(gpu: str) -> str:
    """Convert 'A100:4' → '{A100: 4}'."""
    if ":" in gpu:
        kind, count = gpu.split(":", 1)
        return f"{{{kind}: {count}}}"
    return f"{{{gpu}: 1}}"


def _port_for_engine(engine: ServingEngine) -> int:
    return 8080


class SkyPilotProvider(BaseProvider):
    """Cloud provider backed by SkyPilot / SkyServe."""

    def __init__(self) -> None:
        self._sky_available = self._check_sky()

    def _check_sky(self) -> bool:
        try:
            import sky  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "skypilot is not installed. Install with: pip install 'flare-deploy[aws]'"
            )
            return False

    def _require_sky(self) -> None:
        if not self._sky_available:
            raise ImportError(
                "skypilot is required for deployment. "
                "Install with: pip install 'flare-deploy[aws]' (or gcp/azure/kubernetes)"
            )

    # ------------------------------------------------------------------
    # Task YAML generation
    # ------------------------------------------------------------------

    def _render_task_yaml(
        self,
        spec: ModelSpec,
        entry: DeploymentEntry,
        defaults: GlobalDefaults,
    ) -> str:
        """Render a SkyPilot task YAML string from spec + deployment entry."""
        gpu = entry.gpu or spec.default_gpu
        accelerators = _gpu_spec_to_sky_accelerators(gpu)
        idle_str = entry.resolved_idle_timeout(defaults)
        idle_seconds = _parse_idle_timeout_to_seconds(idle_str)
        port = _port_for_engine(spec.serving.engine)
        min_rep = entry.resolved_min_replicas(defaults)
        max_rep = entry.resolved_max_replicas(defaults)

        ctx: dict = {
            "name": entry.name,
            "gpu_spec": accelerators,
            "disk_gb": spec.disk_gb,
            "port": port,
            "startup_time_seconds": spec.startup_time_seconds,
            "min_replicas": min_rep,
            "max_replicas": max_rep,
            "idle_timeout_seconds": idle_seconds,
            "dtype": spec.serving.dtype,
            "max_model_len": spec.serving.max_model_len or spec.serving.context_length,
            "gpu_memory_utilization": spec.serving.gpu_memory_utilization,
            "tensor_parallel": spec.serving.tensor_parallel,
            "extra_args": spec.serving.extra_args,
            "requires_hf_token": spec.requires_hf_token,
        }

        if spec.is_gguf:
            assert spec.gguf is not None
            ctx.update(
                {
                    "gguf_repo_id": spec.gguf.repo_id,
                    "gguf_filename": spec.gguf.filename,
                    "n_gpu_layers": spec.serving.n_gpu_layers,
                    "context_length": spec.serving.context_length,
                    "n_threads": spec.serving.n_threads,
                }
            )
            template_str = _LLAMACPP_TASK_TEMPLATE
        elif spec.serving.engine == ServingEngine.SGLANG:
            ctx["hf_model_id"] = spec.hf_model_id
            template_str = _SGLANG_TASK_TEMPLATE
        else:
            ctx["hf_model_id"] = spec.hf_model_id
            template_str = _VLLM_TASK_TEMPLATE

        tmpl = _JINJA_ENV.from_string(template_str)
        return tmpl.render(**ctx)

    def _task_from_yaml_str(self, yaml_str: str):  # type: ignore[return]
        """Parse a YAML string into a sky.Task object."""
        import sky
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            tmp.write(yaml_str)
            tmp_path = tmp.name

        try:
            task = sky.Task.from_yaml(tmp_path)
        finally:
            os.unlink(tmp_path)
        return task

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deploy(
        self,
        spec: ModelSpec,
        entry: DeploymentEntry,
        defaults: GlobalDefaults,
    ) -> str:
        self._require_sky()
        import sky

        yaml_str = self._render_task_yaml(spec, entry, defaults)
        logger.debug("SkyPilot task YAML:\n%s", yaml_str)
        task = self._task_from_yaml_str(yaml_str)
        sky.serve.up(task, service_name=entry.name)
        return entry.name

    def get_status(self, service_name: str) -> DeploymentInfo:
        self._require_sky()
        import sky

        try:
            statuses = sky.serve.status(service_names=[service_name])
        except Exception as exc:
            logger.warning("Failed to get status for %s: %s", service_name, exc)
            return DeploymentInfo(name=service_name, state=DeploymentState.UNKNOWN.value)

        if not statuses:
            return DeploymentInfo(name=service_name, state=DeploymentState.OFF.value)

        svc = statuses[0]
        sky_status = svc.get("status", {})
        status_str = sky_status.name if hasattr(sky_status, "name") else str(sky_status)
        state = skypilot_status_to_state(status_str)

        replicas = svc.get("replica_info", [])
        ready = sum(1 for r in replicas if r.get("status", "") == "READY")

        endpoint = svc.get("endpoint")
        if endpoint:
            endpoint = f"http://{endpoint}"

        return DeploymentInfo(
            name=service_name,
            state=state.value,
            replicas_ready=ready,
            replicas_total=len(replicas),
            endpoint=endpoint,
        )

    def list_deployments(self) -> list[DeploymentInfo]:
        self._require_sky()
        import sky

        try:
            statuses = sky.serve.status()
        except Exception as exc:
            logger.warning("Failed to list deployments: %s", exc)
            return []

        results = []
        for svc in statuses:
            name = svc.get("name", "unknown")
            results.append(self.get_status(name))
        return results

    def scale_up(self, service_name: str) -> None:
        self._require_sky()
        import sky

        logger.info("Waking up service: %s", service_name)
        try:
            sky.serve.update(service_name, min_replicas=1)
        except Exception as exc:
            logger.error("Failed to scale up %s: %s", service_name, exc)
            raise

    def stop(self, service_name: str) -> None:
        self._require_sky()
        import sky

        logger.info("Stopping (scale-to-zero) service: %s", service_name)
        sky.serve.update(service_name, min_replicas=0, max_replicas=0)

    def teardown(self, service_name: str) -> None:
        self._require_sky()
        import sky

        logger.info("Tearing down service: %s", service_name)
        sky.serve.down(service_name)

    def stream_logs(self, service_name: str, replica: int = 0) -> Iterator[str]:
        self._require_sky()
        import sky

        try:
            for line in sky.serve.tail_logs(service_name, target="replica", replica_id=replica, follow=True):
                yield line
        except Exception as exc:
            yield f"[error] {exc}"

    # ------------------------------------------------------------------
    # Batch (Phase 2)
    # ------------------------------------------------------------------

    def submit_batch_job(self, spec: ModelSpec, job: BatchJobSpec) -> str:
        self._require_sky()
        import sky
        import uuid

        gpu = job.gpu or spec.default_gpu
        accelerators = _gpu_spec_to_sky_accelerators(gpu)
        job_name = f"{spec.name}-batch-{uuid.uuid4().hex[:8]}"

        tmpl = _JINJA_ENV.from_string(_BATCH_TASK_TEMPLATE)
        yaml_str = tmpl.render(
            name=job_name,
            gpu_spec=accelerators,
            disk_gb=spec.disk_gb,
            hf_model_id=spec.hf_model_id or "",
            requires_hf_token=spec.requires_hf_token,
            input_path=job.input_path,
            output_path=job.output_path,
            use_spot=str(job.use_spot).lower(),
        )

        task = self._task_from_yaml_str(yaml_str)
        sky.jobs.launch(task, name=job_name, retry_until_up=True)
        return job_name

    def get_batch_job_status(self, job_id: str) -> dict:
        self._require_sky()
        import sky

        jobs = sky.jobs.queue(job_name=job_id)
        if not jobs:
            return {"job_id": job_id, "status": "UNKNOWN"}
        return {"job_id": job_id, **jobs[0]}

    def list_batch_jobs(self) -> list[dict]:
        self._require_sky()
        import sky

        try:
            return sky.jobs.queue() or []
        except Exception as exc:
            logger.warning("Failed to list batch jobs: %s", exc)
            return []


# Module-level singleton
_provider: SkyPilotProvider | None = None


def get_provider() -> SkyPilotProvider:
    global _provider
    if _provider is None:
        _provider = SkyPilotProvider()
    return _provider
