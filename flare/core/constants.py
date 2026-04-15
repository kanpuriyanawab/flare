"""Global constants for Flare."""

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent.parent.parent

# Registry directory (repo root / registry)
REGISTRY_DIR = PACKAGE_ROOT / "registry"

# Default flare state directory
FLARE_STATE_DIR = Path.home() / ".flare"
FLARE_DB_PATH = FLARE_STATE_DIR / "flare.db"
FLARE_CONFIG_PATH = FLARE_STATE_DIR / "config.yaml"

# Gateway defaults
GATEWAY_DEFAULT_PORT = 8080
GATEWAY_DEFAULT_HOST = "0.0.0.0"
GATEWAY_POLL_INTERVAL_SECONDS = 30

# Serving
DEFAULT_VLLM_PORT = 8080
DEFAULT_LLAMACPP_PORT = 8080
DEFAULT_SGLANG_PORT = 8080

# OpenAI-compat paths
OPENAI_CHAT_PATH = "/v1/chat/completions"
OPENAI_COMPLETIONS_PATH = "/v1/completions"
OPENAI_MODELS_PATH = "/v1/models"
OPENAI_EMBEDDINGS_PATH = "/v1/embeddings"

# Idle timeout parsing
IDLE_TIMEOUT_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# GPU hourly cost estimates (USD/hr) for cost tracking
GPU_HOURLY_COSTS: dict[str, float] = {
    "T4": 0.35,
    "L4": 0.54,
    "A10G": 1.00,
    "A10": 1.00,
    "A100": 3.21,
    "A100-80GB": 4.10,
    "H100": 8.00,
    "H200": 10.00,
    "B100": 12.00,
    "V100": 2.48,
    "P100": 1.46,
}

# OpenAI token pricing (per 1M tokens, output) for savings comparison
OPENAI_TOKEN_PRICING: dict[str, float] = {
    "gpt-4o": 15.0,
    "gpt-4-turbo": 30.0,
    "gpt-3.5-turbo": 1.5,
}
DEFAULT_OPENAI_COMPARISON_MODEL = "gpt-4o"
