"""Deployment state machine for Flare."""

from __future__ import annotations

from enum import Enum
from typing import Optional


class DeploymentState(str, Enum):
    """States a model deployment can be in."""

    OFF = "OFF"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    SLEEPING = "SLEEPING"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_active(self) -> bool:
        """True if the deployment is consuming GPU resources."""
        return self in (DeploymentState.PROVISIONING, DeploymentState.RUNNING)

    @property
    def is_routable(self) -> bool:
        """True if the gateway can forward requests directly."""
        return self == DeploymentState.RUNNING

    @property
    def needs_wakeup(self) -> bool:
        """True if a new request should trigger a scale-up."""
        return self in (DeploymentState.SLEEPING, DeploymentState.OFF)

    @property
    def display_color(self) -> str:
        """Rich color string for terminal display."""
        colors = {
            DeploymentState.OFF: "dim",
            DeploymentState.PROVISIONING: "yellow",
            DeploymentState.RUNNING: "green",
            DeploymentState.SLEEPING: "blue",
            DeploymentState.FAILED: "red",
            DeploymentState.UNKNOWN: "dim",
        }
        return colors[self]

    def __str__(self) -> str:
        return self.value


# Valid state transitions
VALID_TRANSITIONS: dict[DeploymentState, list[DeploymentState]] = {
    DeploymentState.OFF: [DeploymentState.PROVISIONING],
    DeploymentState.PROVISIONING: [
        DeploymentState.RUNNING,
        DeploymentState.FAILED,
    ],
    DeploymentState.RUNNING: [
        DeploymentState.SLEEPING,
        DeploymentState.FAILED,
        DeploymentState.OFF,
    ],
    DeploymentState.SLEEPING: [
        DeploymentState.PROVISIONING,
        DeploymentState.OFF,
    ],
    DeploymentState.FAILED: [
        DeploymentState.OFF,
        DeploymentState.PROVISIONING,
    ],
    DeploymentState.UNKNOWN: list(DeploymentState),
}


def can_transition(from_state: DeploymentState, to_state: DeploymentState) -> bool:
    """Check whether a state transition is valid."""
    return to_state in VALID_TRANSITIONS.get(from_state, [])


def skypilot_status_to_state(sky_status: Optional[str]) -> DeploymentState:
    """Convert a SkyPilot service status string to a DeploymentState."""
    if sky_status is None:
        return DeploymentState.OFF
    mapping = {
        "READY": DeploymentState.RUNNING,
        "CONTROLLER_INIT": DeploymentState.PROVISIONING,
        "REPLICA_INIT": DeploymentState.PROVISIONING,
        "CONTROLLER_FAILED": DeploymentState.FAILED,
        "FAILED": DeploymentState.FAILED,
        "SHUTTING_DOWN": DeploymentState.OFF,
        "NO_REPLICA": DeploymentState.SLEEPING,
    }
    return mapping.get(sky_status.upper(), DeploymentState.UNKNOWN)
