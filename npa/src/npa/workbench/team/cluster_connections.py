"""Prepare portable Kubernetes connections for a server hosted on another cloud."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from .enrollment import kubectl, verify_enrollment
from .errors import BackendError, TeamError


def read_connection(binding, server_directory: Path) -> dict:
    """Flatten one selected connection and relocate the AWS EKS credential plugin.

    Args:
        binding: Explicit administrator-selected Kubernetes connection.
        server_directory: Private configuration mount in the server containers.
    Returns:
        A minified kubeconfig; credentials must never be printed or sent to jobs.
    Raises:
        BackendError: The selected connection cannot be read or is malformed.
    """
    result = kubectl(
        binding, "config", "view", "--raw", "--flatten", "--minify", "-o", "json"
    )
    if result.returncode:
        raise BackendError("cannot export the selected cluster connection")
    try:
        connection = json.loads(result.stdout)
        for kind in ("clusters", "contexts", "users"):
            if len(connection[kind]) != 1:
                raise ValueError("connection must select exactly one context")
        user = connection["users"][0]["user"]
        _relocate_eks(user, server_directory)
        return connection
    except (ValueError, KeyError, TypeError) as exc:
        raise BackendError("selected cluster connection is invalid") from exc


def _relocate_eks(user, server_directory):
    plugin = user.get("exec", {})
    command = plugin.get("command", "")
    if Path(command).name != "aws":
        return
    arguments = plugin.get("args", [])
    if "eks" not in arguments or "get-token" not in arguments:
        raise ValueError("AWS kubeconfig must use eks get-token")
    environment = {item["name"]: item["value"] for item in plugin.get("env", [])}
    if "--profile" not in arguments and not any(
        arg.startswith("--profile=") for arg in arguments
    ):
        arguments = ["--profile", environment.get("AWS_PROFILE", "default"), *arguments]
    # Export a credential reference, never an operator's inline AWS access keys.
    allowed = ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION")
    environment = {key: environment[key] for key in allowed if key in environment}
    environment.update(
        AWS_CONFIG_FILE=str(server_directory / "aws-config"),
        AWS_SHARED_CREDENTIALS_FILE=str(server_directory / "aws-credentials"),
        AWS_EC2_METADATA_DISABLED="true",
    )
    plugin.update(
        command="/opt/sky/bin/aws",
        args=arguments,
        interactiveMode="Never",
        env=[{"name": key, "value": value} for key, value in environment.items()],
    )


def export_connections(
    config, output_dir: Path, *, server_directory=Path("/etc/npa-team")
):
    """Export matching gateway and scheduler connections without cloud provisioning.

    Args:
        config: Operator policy containing explicitly selected existing clusters.
        output_dir: New private directory for configuration Secret input files.
        server_directory: Absolute configuration mount inside both server containers.
    Returns:
        File names and cluster aliases; never credentials or provider identifiers.
    Raises:
        TeamError, BackendError, OSError: Selection, export, or private writes fail.
    """
    from .deployment import allocations, _write

    if not server_directory.is_absolute():
        raise TeamError("server configuration directory must be absolute")
    bindings = list(allocations(config))
    if not bindings:
        raise TeamError("at least one personal cluster allocation is required")
    connections = {}
    for binding in bindings:
        if binding.cluster not in connections:
            connections[binding.cluster] = read_connection(binding, server_directory)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    policy = config.model_dump(mode="json")
    files = _write_connections(connections, policy, output_dir, server_directory)
    merged = _scheduler_connections(bindings, connections)
    _write(output_dir / "sky-kubeconfig.yaml", yaml.safe_dump(merged))
    _write(output_dir / "team.yaml", yaml.safe_dump(policy, sort_keys=False))
    return {
        "files": files + ["sky-kubeconfig.yaml", "team.yaml"],
        "clusters": list(connections),
    }


def _scheduler_connections(bindings, connections):
    from .deployment import _append_contexts

    merged = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [],
        "users": [],
        "contexts": [],
    }
    for binding in bindings:
        _append_contexts(merged, connections[binding.cluster], binding)
    merged["current-context"] = merged["contexts"][0]["name"]
    return merged


def _write_connections(connections, policy, output_dir, server_directory):
    from .deployment import _write

    files = []
    for name, source in connections.items():
        connection = copy.deepcopy(source)
        filename = f"cluster-{name}.yaml"
        _write(output_dir / filename, yaml.safe_dump(connection))
        policy["clusters"][name] = {
            "context": connection["contexts"][0]["name"],
            "kubeconfig": str(server_directory / filename),
        }
        files.append(filename)
    return files


def check_clusters(config):
    """Check configured cluster access and execution boundaries from the server host.

    Args:
        config: Server policy with mounted credentials and authorized allocations.
    Returns:
        Per-cluster verified allocation counts, without provider identifiers.
    Raises:
        BackendError: Connectivity, authentication, permissions, or enrollment fails.
    """
    from .deployment import allocations

    verified = {}
    for binding in allocations(config):
        verify_enrollment(binding)
        verified[binding.cluster] = verified.get(binding.cluster, 0) + 1
    if not verified:
        raise BackendError("at least one personal cluster allocation is required")
    return {"status": "verified", "allocations": verified}
