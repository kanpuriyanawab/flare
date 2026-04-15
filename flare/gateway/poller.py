"""Background poller: tracks SkyServe state and replays queued requests."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

from flare.core.state import DeploymentState, skypilot_status_to_state
from flare.gateway.queue.base import BaseQueue, RequestStatus

if TYPE_CHECKING:
    from flare.gateway.config import GatewayConfig

logger = logging.getLogger(__name__)


class DeploymentPoller:
    """Polls SkyPilot for deployment state and replays queued requests."""

    def __init__(self, config: "GatewayConfig", queue: BaseQueue) -> None:
        self._config = config
        self._queue = queue
        # model_name → (endpoint, state)
        self._state: dict[str, tuple[str | None, DeploymentState]] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        # Phase 2: pre-warm schedule tracking
        self._next_prewarm: dict[str, datetime] = {}

    def get_state(self, model_name: str) -> DeploymentState:
        if model_name in self._state:
            return self._state[model_name][1]
        return DeploymentState.UNKNOWN

    def get_endpoint(self, model_name: str) -> str | None:
        if model_name in self._state:
            return self._state[model_name][0]
        return None

    def update_state(self, model_name: str, state: DeploymentState, endpoint: str | None) -> None:
        self._state[model_name] = (endpoint, state)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="flare-poller")
        logger.info("Deployment poller started (interval=%ds)", self._config.poll_interval_seconds)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.warning("Poller error: %s", exc, exc_info=True)
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def _poll_once(self) -> None:
        """Query SkyPilot for all service states."""
        try:
            import sky
            statuses = sky.serve.status()
        except ImportError:
            logger.debug("skypilot not installed — skipping poll")
            return
        except Exception as exc:
            logger.warning("SkyPilot status query failed: %s", exc)
            return

        # Build name → (endpoint, state) from SkyPilot response
        for svc in statuses:
            name = svc.get("name", "")
            if not name:
                continue

            sky_status = svc.get("status", {})
            status_str = sky_status.name if hasattr(sky_status, "name") else str(sky_status)
            new_state = skypilot_status_to_state(status_str)
            endpoint = svc.get("endpoint")

            old_state = self._state.get(name, (None, DeploymentState.UNKNOWN))[1]
            self._state[name] = (endpoint, new_state)

            if new_state == DeploymentState.RUNNING and old_state != DeploymentState.RUNNING:
                logger.info("Model %s transitioned to RUNNING — replaying queued requests", name)
                await self._replay_queued(name, endpoint)

    async def _replay_queued(self, model_name: str, endpoint: str | None) -> None:
        """Replay all buffered requests for a model now that it's RUNNING."""
        if not endpoint:
            logger.warning("Model %s is RUNNING but endpoint is unknown — skipping replay", model_name)
            return

        pending = await self._queue.list_pending(model_name)
        if not pending:
            return

        logger.info("Replaying %d queued requests for %s → %s", len(pending), model_name, endpoint)

        async with httpx.AsyncClient(timeout=self._config.request_timeout_seconds) as client:
            for req in pending:
                await self._queue.update_status(req.request_id, RequestStatus.REPLAYING)
                try:
                    target_url = f"{endpoint.rstrip('/')}{req.path}"
                    # Strip hop-by-hop headers that shouldn't be forwarded
                    forward_headers = {
                        k: v for k, v in req.headers.items()
                        if k.lower() not in ("host", "content-length", "transfer-encoding")
                    }
                    resp = await client.request(
                        method=req.method,
                        url=target_url,
                        headers=forward_headers,
                        content=req.body,
                    )
                    await self._queue.update_status(
                        req.request_id,
                        RequestStatus.COMPLETE,
                        response_status=resp.status_code,
                        response_headers=dict(resp.headers),
                        response_body=resp.content,
                    )
                    logger.debug("Replayed request %s → %d", req.request_id, resp.status_code)
                except Exception as exc:
                    logger.error("Failed to replay request %s: %s", req.request_id, exc)
                    await self._queue.update_status(
                        req.request_id,
                        RequestStatus.FAILED,
                        error=str(exc),
                    )

    async def trigger_wakeup(self, model_name: str) -> None:
        """Trigger a scale-up for a sleeping model."""
        try:
            import sky
            sky.serve.update(model_name, min_replicas=1)
            await self._queue.mark_model_waking(model_name)
            logger.info("Wake-up triggered for model: %s", model_name)
        except Exception as exc:
            logger.error("Failed to wake up %s: %s", model_name, exc)
