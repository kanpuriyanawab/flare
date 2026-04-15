"""Redis-backed request queue for HA gateway deployments."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from flare.gateway.queue.base import BaseQueue, QueuedRequest, RequestStatus

logger = logging.getLogger(__name__)

_QUEUE_PREFIX = "flare:queue:"
_REQUEST_PREFIX = "flare:request:"


class RedisQueue(BaseQueue):
    """Redis-backed queue. Use for multi-node / HA gateway deployments.

    Requires: pip install redis (included in flare-deploy[redis])
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._url = redis_url
        self._redis: Optional[object] = None

    async def initialize(self) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis package is required for HA mode. "
                "Install with: pip install 'flare-deploy[redis]'"
            )
        self._redis = aioredis.from_url(self._url, decode_responses=False)
        logger.info("Redis queue connected: %s", self._url)

    def _r(self):
        if self._redis is None:
            raise RuntimeError("RedisQueue not initialized. Call initialize() first.")
        return self._redis

    async def enqueue(self, request: QueuedRequest) -> None:
        data = request.model_dump_json().encode()
        key = f"{_REQUEST_PREFIX}{request.request_id}"
        queue_key = f"{_QUEUE_PREFIX}{request.model_name}"
        r = self._r()
        await r.set(key, data, ex=86400)  # 24hr TTL
        await r.rpush(queue_key, request.request_id)

    async def get(self, request_id: str) -> Optional[QueuedRequest]:
        r = self._r()
        data = await r.get(f"{_REQUEST_PREFIX}{request_id}")
        if data is None:
            return None
        return QueuedRequest.model_validate_json(data)

    async def update_status(
        self,
        request_id: str,
        status: RequestStatus,
        *,
        response_status: Optional[int] = None,
        response_headers: Optional[dict] = None,
        response_body: Optional[bytes] = None,
        error: Optional[str] = None,
    ) -> None:
        req = await self.get(request_id)
        if req is None:
            return
        req.status = status
        if response_status is not None:
            req.response_status = response_status
        if response_headers is not None:
            req.response_headers = response_headers
        if response_body is not None:
            req.response_body = response_body
        if error is not None:
            req.error = error
        if status in (RequestStatus.COMPLETE, RequestStatus.FAILED):
            req.completed_at = datetime.utcnow()

        r = self._r()
        await r.set(
            f"{_REQUEST_PREFIX}{request_id}",
            req.model_dump_json().encode(),
            ex=86400,
        )

    async def list_pending(self, model_name: str) -> list[QueuedRequest]:
        r = self._r()
        queue_key = f"{_QUEUE_PREFIX}{model_name}"
        request_ids = await r.lrange(queue_key, 0, -1)
        results = []
        for rid in request_ids:
            req = await self.get(rid.decode() if isinstance(rid, bytes) else rid)
            if req and req.status in (RequestStatus.QUEUED, RequestStatus.WAKING):
                results.append(req)
        return results

    async def mark_model_waking(self, model_name: str) -> None:
        pending = await self.list_pending(model_name)
        for req in pending:
            if req.status == RequestStatus.QUEUED:
                await self.update_status(req.request_id, RequestStatus.WAKING)
