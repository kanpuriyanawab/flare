"""GET /v1/models — OpenAI-compatible model list."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from flare.gateway.poller import DeploymentPoller

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """Return a list of deployed models in OpenAI format."""
    poller: "DeploymentPoller" = request.app.state.poller

    # Collect all known model names from the poller's state
    model_objects = []
    for name, (endpoint, state) in poller._state.items():
        model_objects.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "flare",
            "permission": [],
            "root": name,
            "parent": None,
            "metadata": {
                "state": state.value,
                "endpoint": endpoint,
            },
        })

    return JSONResponse({
        "object": "list",
        "data": model_objects,
    })
