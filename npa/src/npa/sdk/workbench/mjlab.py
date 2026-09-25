"""Python clients for MJLab's shared runtime and authenticated service."""

from npa.workbench.mjlab.client import invoke
from npa.workbench.mjlab.deployment import DeployRequest, deploy
from npa.workbench.mjlab.schemas import EvalRequest, ExportRequest, TrainRequest


def train(request: TrainRequest, **connection) -> dict:
    """Train or resume an MJLab policy.

    Args:
        request: Validated training request.
        connection: Optional endpoint, token_env and dry_run settings.
    Returns:
        Training manifest or plan.
    Raises:
        MjlabError: Runtime or service failure.
    """
    return invoke("train", request, **connection)


def eval(request: EvalRequest, **connection) -> dict:
    """Measure a native checkpoint in the selected MJLab task.

    Args:
        request: Validated evaluation request.
        connection: Optional endpoint, token_env and dry_run settings.
    Returns:
        Measured evaluation report or plan.
    Raises:
        MjlabError: Runtime or service failure.
    """
    return invoke("eval", request, **connection)


def export(request: ExportRequest, **connection) -> dict:
    """Export a native checkpoint as ONNX.

    Args:
        request: Validated export request.
        connection: Optional endpoint, token_env and dry_run settings.
    Returns:
        Export manifest or plan.
    Raises:
        MjlabError: Runtime or service failure.
    """
    return invoke("export", request, **connection)


def list(**connection) -> dict:
    """List tasks from the installed upstream registry.

    Args:
        connection: Optional endpoint and token_env settings.
    Returns:
        Registered task IDs.
    Raises:
        MjlabError: Runtime or service failure.
    """
    return invoke("list", **connection)


def status(**connection) -> dict:
    """Report local installation or remote service status.

    Args:
        connection: Optional endpoint and token_env settings.
    Returns:
        Installation versions and, remotely, busy state.
    Raises:
        MjlabError: Service failure.
    """
    return invoke("status", **connection)


def system_info(**connection) -> dict:
    """Inspect installed simulator dependencies.

    Args:
        connection: Optional endpoint and token_env settings.
    Returns:
        Dependency version inventory.
    Raises:
        MjlabError: Service failure.
    """
    return invoke("system-info", **connection)


__all__ = [
    "DeployRequest",
    "deploy",
    "EvalRequest",
    "ExportRequest",
    "TrainRequest",
    "train",
    "eval",
    "export",
    "list",
    "status",
    "system_info",
]
