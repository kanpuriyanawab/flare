"""FlareClient — auto-wake Python SDK client."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)


class FlareClient:
    """OpenAI-compatible client that auto-handles cold-start 202 responses.

    Usage::

        from flare.sdk import FlareClient

        client = FlareClient(base_url="http://localhost:8080/v1")
        response = client.chat("qwen3-8b", [{"role": "user", "content": "Hello!"}])
        print(response["choices"][0]["message"]["content"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        timeout: int = 600,
        poll_interval: float = 5.0,
        max_wait: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._max_wait = max_wait

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Send a chat completion request (auto-wakes sleeping models).

        Args:
            model: Registry model name (e.g. 'qwen3-8b').
            messages: OpenAI-format message list.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            stream: If True, returns a streaming iterator.
            **kwargs: Extra fields forwarded to the API.

        Returns:
            OpenAI-compatible response dict.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        return self._request_with_retry("/chat/completions", payload)

    def complete(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict:
        """Send a text completion request."""
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        return self._request_with_retry("/completions", payload)

    def embed(self, model: str, input: list[str] | str, **kwargs: Any) -> dict:
        """Generate embeddings."""
        payload = {"model": model, "input": input, **kwargs}
        return self._request_with_retry("/embeddings", payload)

    def models(self) -> list[dict]:
        """List available models."""
        with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
            resp = client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            return resp.json().get("data", [])

    # ------------------------------------------------------------------
    # Core retry / polling logic
    # ------------------------------------------------------------------

    def _request_with_retry(self, path: str, payload: dict) -> dict:
        """POST to the gateway, handling 202 cold-start queuing automatically."""
        url = f"{self.base_url}{path}"

        with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
            resp = client.post(url, json=payload)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 202:
                # Cold start — poll until complete
                body = resp.json()
                request_id = body.get("request_id")
                wait = body.get("estimated_wait_seconds", 300)
                logger.info(
                    "Model '%s' is waking up (request_id=%s, est. %ds). Polling...",
                    payload.get("model"),
                    request_id,
                    wait,
                )
                return self._poll_until_complete(client, request_id)

            # Error
            resp.raise_for_status()
            return resp.json()  # unreachable but satisfies type checker

    def _poll_until_complete(self, client: httpx.Client, request_id: str) -> dict:
        """Poll /v1/requests/{id} until status == complete."""
        poll_url = f"{self.base_url}/requests/{request_id}"
        deadline = time.monotonic() + self._max_wait

        while time.monotonic() < deadline:
            time.sleep(self._poll_interval)
            resp = client.get(poll_url)
            if resp.status_code == 404:
                raise RuntimeError(f"Request {request_id} not found on gateway.")

            data = resp.json()
            status = data.get("status")

            if status == "complete":
                # The response body is the original model response
                return data if "choices" in data else resp.json()

            if status == "failed":
                raise RuntimeError(
                    f"Request {request_id} failed: {data.get('error', 'Unknown error')}"
                )

            logger.debug(
                "Request %s still pending (est. %ds remaining)",
                request_id,
                data.get("estimated_wait_seconds", "?"),
            )

        raise TimeoutError(
            f"Request {request_id} did not complete within {self._max_wait}s. "
            "The model may still be starting. Poll manually with the request_id."
        )

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def achat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict:
        """Async chat completion."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        return await self._async_request_with_retry("/chat/completions", payload)

    async def _async_request_with_retry(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
            resp = await client.post(url, json=payload)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 202:
                body = resp.json()
                request_id = body.get("request_id")
                wait = body.get("estimated_wait_seconds", 300)
                logger.info("Polling request %s (est. %ds)...", request_id, wait)
                return await self._async_poll_until_complete(client, request_id)

            resp.raise_for_status()
            return {}

    async def _async_poll_until_complete(
        self, client: httpx.AsyncClient, request_id: str
    ) -> dict:
        poll_url = f"{self.base_url}/requests/{request_id}"
        deadline = asyncio.get_event_loop().time() + self._max_wait

        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(self._poll_interval)
            resp = await client.get(poll_url)
            data = resp.json()
            status = data.get("status")
            if status == "complete":
                return data
            if status == "failed":
                raise RuntimeError(f"Request failed: {data.get('error')}")

        raise TimeoutError(f"Request {request_id} timed out after {self._max_wait}s.")
