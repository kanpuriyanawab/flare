"""Gateway configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    db_path: str = str(Path.home() / ".flare" / "gateway.db")
    redis_url: Optional[str] = None
    poll_interval_seconds: int = 30
    request_timeout_seconds: int = 600
    require_api_key: bool = False

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        return cls(
            host=os.getenv("FLARE_GATEWAY_HOST", "0.0.0.0"),
            port=int(os.getenv("FLARE_GATEWAY_PORT", "8080")),
            db_path=os.getenv("FLARE_DB_PATH", str(Path.home() / ".flare" / "gateway.db")),
            redis_url=os.getenv("FLARE_REDIS_URL"),
            poll_interval_seconds=int(os.getenv("FLARE_POLL_INTERVAL", "30")),
            require_api_key=os.getenv("FLARE_REQUIRE_API_KEY", "false").lower() == "true",
        )
