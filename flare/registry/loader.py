"""Registry loader: discovers and validates model YAML files from the registry/ directory."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import yaml
from pydantic import ValidationError

from flare.core.config import ModelSpec
from flare.core.constants import REGISTRY_DIR
from flare.core.exceptions import ModelNotFoundError, RegistryError

logger = logging.getLogger(__name__)


def _iter_yaml_files(registry_dir: Path) -> Iterator[Path]:
    """Yield all .yaml files recursively under registry_dir."""
    for path in sorted(registry_dir.rglob("*.yaml")):
        yield path


def _load_spec_from_file(path: Path) -> ModelSpec:
    """Parse and validate a single registry YAML file."""
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise RegistryError(f"YAML parse error in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RegistryError(f"Registry file {path} must contain a YAML mapping")

    try:
        return ModelSpec.model_validate(raw)
    except ValidationError as exc:
        raise RegistryError(f"Invalid model spec in {path}:\n{exc}") from exc


class Registry:
    """In-memory registry of all validated ModelSpec objects."""

    def __init__(self, registry_dir: Path = REGISTRY_DIR) -> None:
        self._dir = registry_dir
        self._models: dict[str, ModelSpec] = {}
        self._loaded = False

    def load(self, *, strict: bool = False) -> None:
        """Load all model specs from the registry directory.

        Args:
            strict: If True, raise on any invalid file. If False, skip and warn.
        """
        if not self._dir.exists():
            raise RegistryError(
                f"Registry directory not found: {self._dir}. "
                "Ensure you are running from the flare repo root."
            )

        self._models = {}
        errors: list[str] = []

        for yaml_file in _iter_yaml_files(self._dir):
            try:
                spec = _load_spec_from_file(yaml_file)
                if spec.name in self._models:
                    logger.warning(
                        "Duplicate registry entry '%s' (from %s). Skipping.",
                        spec.name,
                        yaml_file,
                    )
                    continue
                self._models[spec.name] = spec
                logger.debug("Loaded model spec: %s", spec.name)
            except RegistryError as exc:
                if strict:
                    raise
                errors.append(str(exc))
                logger.warning("Skipping %s: %s", yaml_file.name, exc)

        self._loaded = True
        logger.info("Registry loaded: %d models (%d errors)", len(self._models), len(errors))

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get(self, name: str) -> ModelSpec:
        """Return a ModelSpec by name, raising ModelNotFoundError if absent."""
        self._ensure_loaded()
        try:
            return self._models[name]
        except KeyError:
            raise ModelNotFoundError(name)

    def all(self) -> list[ModelSpec]:
        """Return all loaded model specs sorted by family then name."""
        self._ensure_loaded()
        return sorted(self._models.values(), key=lambda m: (m.family, m.name))

    def search(
        self,
        *,
        family: str | None = None,
        tag: str | None = None,
        capability: str | None = None,
        engine: str | None = None,
    ) -> list[ModelSpec]:
        """Filter models by optional criteria."""
        self._ensure_loaded()
        results = list(self._models.values())
        if family:
            results = [m for m in results if m.family.lower() == family.lower()]
        if tag:
            results = [m for m in results if tag.lower() in [t.lower() for t in m.tags]]
        if capability:
            results = [m for m in results if capability.lower() in [c.lower() for c in m.capabilities]]
        if engine:
            results = [m for m in results if m.serving.engine.value == engine.lower()]
        return sorted(results, key=lambda m: (m.family, m.name))

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        self._ensure_loaded()
        return name in self._models


# Module-level singleton
_registry: Registry | None = None


def get_registry() -> Registry:
    """Return the global Registry singleton (lazy-loaded)."""
    global _registry
    if _registry is None:
        _registry = Registry()
        _registry.load()
    return _registry


def reload_registry() -> Registry:
    """Force-reload the global registry (useful after adding new YAML files)."""
    global _registry
    _registry = Registry()
    _registry.load()
    return _registry
