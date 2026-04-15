"""Pydantic models for Flare configuration: registry specs and models.yaml."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ServingEngine(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    LLAMA_CPP = "llama-cpp"  # for GGUF checkpoints


class InfraProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    LAMBDA = "lambda"
    RUNPOD = "runpod"


class DeploymentMode(str, Enum):
    ON_DEMAND = "on-demand"
    BATCH = "batch"


# ---------------------------------------------------------------------------
# Registry model spec
# ---------------------------------------------------------------------------


class GpuSpec(BaseModel):
    """GPU requirements for a registry model."""

    recommended: list[str] = Field(..., description="Preferred GPU configs, e.g. ['A100:4']")
    minimum: list[str] = Field(..., description="Minimum viable GPU configs")
    fallback: list[str] = Field(default_factory=list, description="Fallback GPU configs")


class ServingConfig(BaseModel):
    """How to launch the model server."""

    engine: ServingEngine = ServingEngine.VLLM
    tensor_parallel: int = Field(default=1, ge=1)
    context_length: int = Field(default=4096, ge=512)
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=1.0)
    # llama-cpp specific
    n_gpu_layers: int = -1
    n_threads: int = 8
    # extra vllm/sglang flags appended verbatim
    extra_args: list[str] = Field(default_factory=list)


class GgufSpec(BaseModel):
    """Specification for a GGUF checkpoint."""

    repo_id: str = Field(..., description="HuggingFace repo ID hosting the GGUF file")
    filename: str = Field(..., description="Filename of the .gguf checkpoint in the repo")
    quantization: str = Field(..., description="Quantization label, e.g. Q4_K_M")


class ModelSpec(BaseModel):
    """Full specification of a model in the Flare registry."""

    name: str = Field(..., description="Unique registry identifier, e.g. qwen3-72b")
    display_name: str
    family: str
    version: str
    description: str = ""
    hf_model_id: Optional[str] = None
    gguf: Optional[GgufSpec] = None
    serving: ServingConfig = Field(default_factory=ServingConfig)
    gpus: GpuSpec
    memory_gb: float = Field(..., ge=1)
    disk_gb: int = Field(default=100, ge=1)
    startup_time_seconds: int = Field(default=300, ge=30)
    requires_hf_token: bool = False
    capabilities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_model_source(self) -> "ModelSpec":
        if self.hf_model_id is None and self.gguf is None:
            raise ValueError("Either hf_model_id or gguf must be specified")
        if self.gguf is not None and self.serving.engine != ServingEngine.LLAMA_CPP:
            # Auto-correct engine for GGUF models
            self.serving.engine = ServingEngine.LLAMA_CPP
        return self

    @property
    def is_gguf(self) -> bool:
        return self.gguf is not None

    @property
    def default_gpu(self) -> str:
        """First recommended GPU config."""
        return self.gpus.recommended[0] if self.gpus.recommended else "A10G:1"


# ---------------------------------------------------------------------------
# models.yaml (user config)
# ---------------------------------------------------------------------------


def _parse_idle_timeout(value: str) -> int:
    """Convert '15m', '1h', '30s' → seconds (int)."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    match = re.fullmatch(r"(\d+)([smhd])", value.strip())
    if not match:
        raise ValueError(f"Invalid idle_timeout format: '{value}'. Use e.g. '15m', '1h', '30s'.")
    return int(match.group(1)) * units[match.group(2)]


class GlobalDefaults(BaseModel):
    """Default values applied to all model deployments."""

    idle_timeout: str = "15m"
    min_replicas: int = Field(default=0, ge=0)
    max_replicas: int = Field(default=3, ge=1)

    @field_validator("idle_timeout")
    @classmethod
    def validate_timeout(cls, v: str) -> str:
        _parse_idle_timeout(v)  # validates format
        return v

    @property
    def idle_timeout_seconds(self) -> int:
        return _parse_idle_timeout(self.idle_timeout)


class DeploymentEntry(BaseModel):
    """A single model entry in models.yaml."""

    name: str = Field(..., description="Must match a registry model name")
    mode: DeploymentMode = DeploymentMode.ON_DEMAND
    gpu: Optional[str] = None
    idle_timeout: Optional[str] = None
    min_replicas: Optional[int] = Field(default=None, ge=0)
    max_replicas: Optional[int] = Field(default=None, ge=1)
    # Phase 2: cron pre-warm schedule (e.g. "0 8 * * 1-5")
    schedule: Optional[str] = None

    @field_validator("idle_timeout")
    @classmethod
    def validate_timeout(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            _parse_idle_timeout(v)
        return v

    def resolved_idle_timeout(self, defaults: GlobalDefaults) -> str:
        return self.idle_timeout or defaults.idle_timeout

    def resolved_min_replicas(self, defaults: GlobalDefaults) -> int:
        return self.min_replicas if self.min_replicas is not None else defaults.min_replicas

    def resolved_max_replicas(self, defaults: GlobalDefaults) -> int:
        return self.max_replicas if self.max_replicas is not None else defaults.max_replicas


class FlareConfig(BaseModel):
    """Top-level models.yaml configuration."""

    infra: InfraProvider
    region: Optional[str] = None
    defaults: GlobalDefaults = Field(default_factory=GlobalDefaults)
    models: list[DeploymentEntry] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "FlareConfig":
        import yaml
        from pathlib import Path

        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid models.yaml: expected a mapping at top level")
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Gateway config
# ---------------------------------------------------------------------------


class GatewayConfig(BaseModel):
    """Configuration for the Flare gateway server."""

    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    db_url: str = "sqlite+aiosqlite:///~/.flare/gateway.db"
    redis_url: Optional[str] = None
    poll_interval_seconds: int = 30
    request_timeout_seconds: int = 600
    # API key enforcement (Phase 2)
    require_api_key: bool = False

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        import os

        return cls(
            host=os.getenv("FLARE_GATEWAY_HOST", "0.0.0.0"),
            port=int(os.getenv("FLARE_GATEWAY_PORT", "8080")),
            db_url=os.getenv(
                "FLARE_DB_URL", "sqlite+aiosqlite:///~/.flare/gateway.db"
            ),
            redis_url=os.getenv("FLARE_REDIS_URL"),
            poll_interval_seconds=int(os.getenv("FLARE_POLL_INTERVAL", "30")),
            require_api_key=os.getenv("FLARE_REQUIRE_API_KEY", "false").lower() == "true",
        )


# ---------------------------------------------------------------------------
# Deployment runtime info (returned by provider.get_status)
# ---------------------------------------------------------------------------


class DeploymentInfo(BaseModel):
    """Runtime state of a deployment (returned by the provider)."""

    name: str
    state: str  # DeploymentState value
    replicas_ready: int = 0
    replicas_total: int = 0
    gpu: Optional[str] = None
    endpoint: Optional[str] = None
    cost_per_hour: Optional[float] = None
    uptime_seconds: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Phase 2: Batch job spec
# ---------------------------------------------------------------------------


class BatchJobSpec(BaseModel):
    """Spec for a batch inference job (Phase 2)."""

    model_name: str
    input_path: str = Field(..., description="S3/GCS URI or local path to input JSONL")
    output_path: str = Field(..., description="S3/GCS URI to write results")
    gpu: Optional[str] = None
    use_spot: bool = True
    max_retries: int = 3
    extra_args: dict[str, Any] = Field(default_factory=dict)
