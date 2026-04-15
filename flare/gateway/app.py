"""FastAPI gateway application factory."""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from flare.gateway.config import GatewayConfig
from flare.gateway.poller import DeploymentPoller
from flare.gateway.queue.base import BaseQueue
from flare.gateway.queue.sqlite_queue import SQLiteQueue
from flare.gateway.routes.chat import router as chat_router
from flare.gateway.routes.models_route import router as models_router
from flare.gateway.routes.requests_route import router as requests_router

logger = logging.getLogger(__name__)


def _build_queue(config: GatewayConfig) -> BaseQueue:
    if config.redis_url:
        from flare.gateway.queue.redis_queue import RedisQueue
        return RedisQueue(config.redis_url)
    return SQLiteQueue(db_path=config.db_path)


def create_app(config: Optional[GatewayConfig] = None) -> FastAPI:
    """FastAPI application factory.

    Usage:
        uvicorn flare.gateway.app:create_app --factory --host 0.0.0.0 --port 8080
    """
    if config is None:
        config = GatewayConfig.from_env()

    queue = _build_queue(config)
    poller = DeploymentPoller(config, queue)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await queue.initialize()
        await poller.start()
        logger.info("Flare gateway started on %s:%d", config.host, config.port)
        yield
        # Shutdown
        await poller.stop()
        logger.info("Flare gateway stopped")

    app = FastAPI(
        title="Flare Gateway",
        description="OpenAI-compatible LLM gateway with cold-start queuing",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store shared state
    app.state.gateway_config = config
    app.state.queue = queue
    app.state.poller = poller

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API key middleware (Phase 2)
    if config.require_api_key:
        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next):
            # Skip health check
            if request.url.path in ("/health", "/", "/docs", "/openapi.json"):
                return await call_next(request)

            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse(
                    {"error": {"message": "Missing API key", "type": "auth_error"}},
                    status_code=401,
                )

            api_key = auth[len("Bearer "):]
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Validate key from DB
            is_valid = await _validate_api_key(config.db_path, key_hash)
            if not is_valid:
                return JSONResponse(
                    {"error": {"message": "Invalid API key", "type": "auth_error"}},
                    status_code=401,
                )
            return await call_next(request)

    # Routes
    app.include_router(chat_router)
    app.include_router(models_router)
    app.include_router(requests_router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "Flare Gateway",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app


async def _validate_api_key(db_path: str, key_hash: str) -> bool:
    try:
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT 1 FROM api_keys WHERE key_hash = ? AND is_active = 1", (key_hash,)
            ) as cursor:
                return await cursor.fetchone() is not None
    except Exception:
        return False
