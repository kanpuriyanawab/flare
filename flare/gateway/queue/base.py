"""Abstract interface for the Flare request queue."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class RequestStatus(str, Enum):
    QUEUED = "queued"
    WAKING = "waking"
    REPLAYING = "replaying"
    COMPLETE = "complete"
    FAILED = "failed"


class QueuedRequest(BaseModel):
    """A request that was queued because the model was sleeping."""

    request_id: str
    model_name: str
    path: str                           # e.g. /v1/chat/completions
    method: str                         # POST
    headers: dict[str, str]
    body: bytes
    status: RequestStatus = RequestStatus.QUEUED
    created_at: datetime
    completed_at: Optional[datetime] = None
    response_status: Optional[int] = None
    response_headers: Optional[dict[str, str]] = None
    response_body: Optional[bytes] = None
    error: Optional[str] = None
    estimated_wait_seconds: int = 300

    model_config = {"arbitrary_types_allowed": True}


class BaseQueue(ABC):
    """Abstract persistent queue for cold-start request buffering."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables / connections."""

    @abstractmethod
    async def enqueue(self, request: QueuedRequest) -> None:
        """Store a request in the queue."""

    @abstractmethod
    async def get(self, request_id: str) -> Optional[QueuedRequest]:
        """Retrieve a request by ID."""

    @abstractmethod
    async def update_status(
        self,
        request_id: str,
        status: RequestStatus,
        *,
        response_status: Optional[int] = None,
        response_headers: Optional[dict[str, str]] = None,
        response_body: Optional[bytes] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the status (and optionally the response) of a queued request."""

    @abstractmethod
    async def list_pending(self, model_name: str) -> list[QueuedRequest]:
        """Return all QUEUED or WAKING requests for a given model."""

    @abstractmethod
    async def mark_model_waking(self, model_name: str) -> None:
        """Mark all QUEUED requests for a model as WAKING."""
