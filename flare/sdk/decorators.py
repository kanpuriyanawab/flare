"""@serve decorator for zero-config model serving."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def serve(
    model: str,
    *,
    gpu: Optional[str] = None,
    min_replicas: int = 0,
    max_replicas: int = 3,
    idle_timeout: str = "15m",
    infra: Optional[str] = None,
    gateway_url: str = "http://localhost:8080/v1",
    api_key: Optional[str] = None,
    auto_deploy: bool = True,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator that wires a class to a Flare registry model.

    The decorated class gets a `.client` attribute (FlareClient) and
    a `.chat()` / `.complete()` method that auto-wakes the model.

    Usage::

        from flare.sdk import serve

        @serve(model="qwen3-8b", gpu="L4:1", min_replicas=0)
        class Assistant:
            pass

        assistant = Assistant()
        response = assistant.chat([{"role": "user", "content": "Hello!"}])

    Args:
        model: Registry model name.
        gpu: GPU spec override (e.g. 'A100:4'). Uses registry default if None.
        min_replicas: Minimum replicas (0 = scale to zero).
        max_replicas: Maximum replicas for autoscaling.
        idle_timeout: Scale-to-zero timeout.
        infra: Cloud provider override.
        gateway_url: Flare gateway base URL.
        api_key: Optional API key for the gateway.
        auto_deploy: If True, deploy the model when the class is instantiated
                     and it is not yet deployed.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        original_init = cls.__init__  # type: ignore[misc]

        @functools.wraps(original_init)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)

            from flare.sdk.client import FlareClient
            self.client = FlareClient(
                base_url=gateway_url,
                api_key=api_key,
            )
            self._flare_model = model

            if auto_deploy:
                self._ensure_deployed()

        def _ensure_deployed(self: Any) -> None:
            """Deploy the model if not already running."""
            try:
                from flare.registry.loader import get_registry
                from flare.providers.skypilot import get_provider
                from flare.core.config import DeploymentEntry, GlobalDefaults
                from flare.core.state import DeploymentState

                provider = get_provider()
                info = provider.get_status(self._flare_model)
                state = DeploymentState(info.state)

                if state == DeploymentState.OFF:
                    logger.info(
                        "@serve: auto-deploying model '%s' (gpu=%s)...",
                        self._flare_model,
                        gpu or "default",
                    )
                    registry = get_registry()
                    spec = registry.get(self._flare_model)
                    entry = DeploymentEntry(
                        name=self._flare_model,
                        gpu=gpu,
                        min_replicas=min_replicas,
                        max_replicas=max_replicas,
                        idle_timeout=idle_timeout,
                    )
                    provider.deploy(spec, entry, GlobalDefaults())
                    logger.info("@serve: deployment submitted for '%s'", self._flare_model)
                else:
                    logger.debug(
                        "@serve: model '%s' is already %s", self._flare_model, state.value
                    )
            except Exception as exc:
                logger.warning(
                    "@serve: auto-deploy failed for '%s': %s", self._flare_model, exc
                )

        def chat(self: Any, messages: list[dict], **kwargs: Any) -> dict:
            return self.client.chat(self._flare_model, messages, **kwargs)

        def complete(self: Any, prompt: str, **kwargs: Any) -> dict:
            return self.client.complete(self._flare_model, prompt, **kwargs)

        cls.__init__ = __init__  # type: ignore[misc]
        cls._ensure_deployed = _ensure_deployed
        cls.chat = chat
        cls.complete = complete

        # Attach metadata
        cls._flare_config = {
            "model": model,
            "gpu": gpu,
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "idle_timeout": idle_timeout,
            "gateway_url": gateway_url,
        }

        return cls

    return decorator
