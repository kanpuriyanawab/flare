"""POST /v1/chat/completions — proxy or queue cold-start requests."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

from flare.core.state import DeploymentState
from flare.gateway.queue.base import QueuedRequest, RequestStatus

if TYPE_CHECKING:
    from flare.gateway.poller import DeploymentPoller
    from flare.gateway.queue.base import BaseQueue

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_request(request, "/v1/chat/completions")


@router.post("/v1/completions")
async def completions(request: Request):
    return await _handle_openai_request(request, "/v1/completions")


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    return await _handle_openai_request(request, "/v1/embeddings")


async def _handle_openai_request(request: Request, path: str):
    """Core proxy logic: forward if RUNNING, queue if SLEEPING/OFF."""
    body = await request.body()

    # Extract model name from request body
    model_name: str | None = None
    try:
        payload = json.loads(body)
        model_name = payload.get("model")
    except (json.JSONDecodeError, AttributeError):
        pass

    if not model_name:
        return JSONResponse(
            {"error": {"message": "Missing 'model' field in request body", "type": "invalid_request_error"}},
            status_code=400,
        )

    poller: "DeploymentPoller" = request.app.state.poller
    queue: "BaseQueue" = request.app.state.queue
    config = request.app.state.gateway_config

    state = poller.get_state(model_name)
    endpoint = poller.get_endpoint(model_name)

    # --- RUNNING: proxy directly ---
    if state.is_routable and endpoint:
        return await _proxy_request(request, path, body, endpoint, config.request_timeout_seconds)

    # --- SLEEPING or OFF: queue + trigger wake-up, return 202 ---
    if state.needs_wakeup or state == DeploymentState.UNKNOWN:
        request_id = str(uuid.uuid4())
        headers = dict(request.headers)

        # Look up estimated startup time from registry
        wait_seconds = 300
        try:
            from flare.registry.loader import get_registry
            registry = get_registry()
            if model_name in registry:
                wait_seconds = registry.get(model_name).startup_time_seconds
        except Exception:
            pass

        queued_req = QueuedRequest(
            request_id=request_id,
            model_name=model_name,
            path=path,
            method="POST",
            headers=headers,
            body=body,
            status=RequestStatus.QUEUED,
            created_at=datetime.utcnow(),
            estimated_wait_seconds=wait_seconds,
        )
        await queue.enqueue(queued_req)

        # Trigger wake-up (non-blocking)
        try:
            await poller.trigger_wakeup(model_name)
        except Exception as exc:
            logger.warning("Wake-up trigger failed for %s: %s", model_name, exc)

        poll_url = f"/v1/requests/{request_id}"
        return JSONResponse(
            {
                "request_id": request_id,
                "status": "queued",
                "model": model_name,
                "estimated_wait_seconds": wait_seconds,
                "poll_url": poll_url,
                "message": (
                    f"Model '{model_name}' is waking up. Poll {poll_url} for status. "
                    "Estimated wait: ~{wait_seconds}s."
                ),
            },
            status_code=202,
        )

    # --- PROVISIONING: tell client to retry ---
    if state == DeploymentState.PROVISIONING:
        return JSONResponse(
            {
                "error": {
                    "message": f"Model '{model_name}' is still starting up. Retry in 30s.",
                    "type": "service_unavailable",
                    "code": "model_provisioning",
                }
            },
            status_code=503,
            headers={"Retry-After": "30"},
        )

    # --- FAILED ---
    return JSONResponse(
        {
            "error": {
                "message": f"Model '{model_name}' is in FAILED state. Re-deploy with `flare deploy {model_name}`.",
                "type": "service_unavailable",
                "code": "model_failed",
            }
        },
        status_code=503,
    )


async def _proxy_request(
    request: Request,
    path: str,
    body: bytes,
    endpoint: str,
    timeout: int,
) -> Response:
    """Forward request to the model endpoint and return the response."""
    target_url = f"{endpoint.rstrip('/')}{path}"
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(
            method="POST",
            url=target_url,
            headers=forward_headers,
            content=body,
        )

    # Handle streaming responses
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        async def stream_generator():
            async with httpx.AsyncClient(timeout=timeout) as streaming_client:
                async with streaming_client.stream(
                    "POST", target_url, headers=forward_headers, content=body
                ) as streamed:
                    async for chunk in streamed.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_generator(),
            status_code=resp.status_code,
            media_type="text/event-stream",
        )

    response_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding")
    }
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers,
        media_type=content_type,
    )
