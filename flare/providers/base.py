"""Abstract base class for cloud infrastructure providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from flare.core.config import BatchJobSpec, DeploymentEntry, DeploymentInfo, GlobalDefaults, ModelSpec


class BaseProvider(ABC):
    """Abstract provider: wraps a cloud/orchestration backend."""

    @abstractmethod
    def deploy(
        self,
        spec: ModelSpec,
        entry: DeploymentEntry,
        defaults: GlobalDefaults,
    ) -> str:
        """Deploy a model.

        Returns:
            Service name (usable for subsequent calls).
        """

    @abstractmethod
    def get_status(self, service_name: str) -> DeploymentInfo:
        """Return the current runtime state of a deployment."""

    @abstractmethod
    def list_deployments(self) -> list[DeploymentInfo]:
        """Return all tracked deployments."""

    @abstractmethod
    def scale_up(self, service_name: str) -> None:
        """Wake a sleeping deployment (scale min_replicas to at least 1)."""

    @abstractmethod
    def stop(self, service_name: str) -> None:
        """Scale a deployment to zero (preserve config)."""

    @abstractmethod
    def teardown(self, service_name: str) -> None:
        """Fully remove a deployment and all its resources."""

    @abstractmethod
    def stream_logs(self, service_name: str, replica: int = 0) -> Iterator[str]:
        """Yield log lines from an active deployment."""

    @abstractmethod
    def submit_batch_job(self, spec: ModelSpec, job: BatchJobSpec) -> str:
        """Submit a batch inference job.

        Returns:
            Job ID.
        """

    @abstractmethod
    def get_batch_job_status(self, job_id: str) -> dict:
        """Return status of a batch job."""

    @abstractmethod
    def list_batch_jobs(self) -> list[dict]:
        """Return all batch jobs."""
