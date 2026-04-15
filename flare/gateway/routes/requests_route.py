"""GET /v1/requests/{request_id} — poll a queued cold-start request."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

if TYPE_CHECKING:
    from flare.gateway.queue.base import BaseQueue

router = APIRouter()


@router.get("/v1/requests/{request_id}")
async def poll_request(request_id: str, request: Request):
    """Poll the status of a cold-start queued request.

    Returns:
        - 200 with {status: "pending", estimated_wait: N} while waiting.
        - 200 with {status: "complete", response: {...}} when done.
        - 200 with {status: "failed", error: "..."} on failure.
        - 404 if request_id not found.
    """
    queue: "BaseQueue" = request.app.state.queue
    queued = await queue.get(request_id)

    if queued is None:
        raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found.")

    status = queued.status.value

    if status in ("queued", "waking", "replaying"):
        return JSONResponse({
            "request_id": request_id,
            "status": "pending",
            "model": queued.model_name,
            "estimated_wait_seconds": queued.estimated_wait_seconds,
            "created_at": queued.created_at.isoformat(),
        })

    if status == "complete":
        if queued.response_body is None:
            raise HTTPException(status_code=500, detail="Request completed but response body is missing.")
        # Return the original response
        headers = dict(queued.response_headers or {})
        # Strip transfer-encoding to avoid issues
        headers.pop("transfer-encoding", None)
        headers.pop("content-length", None)
        headers.pop("content-encoding", None)
        return Response(
            content=queued.response_body,
            status_code=queued.response_status or 200,
            headers=headers,
            media_type=headers.get("content-type", "application/json"),
        )

    # failed
    return JSONResponse(
        {
            "request_id": request_id,
            "status": "failed",
            "error": queued.error or "Unknown error",
        },
        status_code=200,
    )
