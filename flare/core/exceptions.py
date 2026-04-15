"""Custom exceptions for Flare."""


class FlareError(Exception):
    """Base exception for all Flare errors."""


class ModelNotFoundError(FlareError):
    """Raised when a model is not found in the registry."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' not found in registry. "
            f"Run `flare catalog` to see available models."
        )
        self.model_name = model_name


class DeploymentNotFoundError(FlareError):
    """Raised when a deployment does not exist."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"No active deployment found for '{model_name}'. "
            f"Run `flare deploy {model_name}` first."
        )
        self.model_name = model_name


class InvalidConfigError(FlareError):
    """Raised when a configuration file is invalid."""


class ProviderError(FlareError):
    """Raised when a cloud provider operation fails."""


class GatewayError(FlareError):
    """Raised when the gateway encounters an error."""


class RegistryError(FlareError):
    """Raised when the registry encounters a loading error."""


class InfraNotInitializedError(FlareError):
    """Raised when flare init has not been run."""

    def __init__(self) -> None:
        super().__init__(
            "Flare is not initialized. Run `flare init --infra <aws|gcp|azure|kubernetes>` first."
        )
