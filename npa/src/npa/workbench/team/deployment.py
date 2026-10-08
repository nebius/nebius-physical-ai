"""Prepare private SkyPilot configuration without uploading operator credentials to jobs."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import shlex
import ipaddress
from urllib.parse import urlsplit
from pathlib import Path

import yaml

from .authorization import ExecutionBinding
from .enrollment import kubectl
from .errors import BackendError
from .manifests import (
    CONTROLLER_ACCOUNT,
    WORKER_ACCOUNT,
    admission_manifests,
    execution_manifests,
)
from .models import execution_name
from .workflow_policy import controller_context, worker_context


def allocations(config):
    """Enumerate explicit personal allocations without inventing group membership.

    Args:
        config: Validated installation configuration.
    Returns:
        Iterator of explicit personal cluster bindings.
    Raises:
        None.
    """
    for workspace, policy in config.workspaces.items():
        for allocation in policy.allocations:
            name = execution_name(config.identity.issuer, workspace, allocation.subject)
            for cluster in allocation.clusters:
                yield ExecutionBinding(
                    workspace, cluster, name, allocation, config.clusters[cluster]
                )


def render_installation(config, output_dir: Path):
    """Write reviewable manifests and private scheduler settings for all enrolled clusters.

    Args:
        config: Validated administrator policy.
        output_dir: New private destination, never an existing directory.
    Returns:
        Generated file paths without credentials or personal identities.
    Raises:
        OSError: The destination exists or cannot be written.
    """
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    grouped = {cluster: admission_manifests() for cluster in config.clusters}
    for binding in allocations(config):
        grouped[binding.cluster].extend(execution_manifests(binding))
    paths = []
    for cluster, documents in grouped.items():
        path = output_dir / f"{cluster}-enrollment.yaml"
        _write(path, yaml.safe_dump_all(documents, sort_keys=False))
        paths.append(str(path))
    path = output_dir / "sky-server.yaml"
    _write(path, yaml.safe_dump(server_config(config), sort_keys=False))
    paths.append(str(path))
    return {"files": paths, "credentials_exported": False}


def export_server_kubeconfig(config, output_path: Path):
    """Export only selected cluster connections into a private scheduler kubeconfig.

    Args:
        config: Trusted cluster enrollment configuration.
        output_path: New private credential file on the server, never in a workload.
    Returns:
        File path without credential contents.
    Raises:
        BackendError, OSError: Selected connections cannot be exported safely.
    """
    merged = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [],
        "users": [],
        "contexts": [],
    }
    for binding in allocations(config):
        response = kubectl(
            binding, "config", "view", "--raw", "--flatten", "--minify", "-o", "json"
        )
        if response.returncode:
            raise BackendError("cannot export the selected cluster connection")
        source = json.loads(response.stdout)
        _append_contexts(merged, source, binding)
    if not merged["contexts"]:
        raise BackendError("at least one personal cluster allocation is required")
    merged["current-context"] = merged["contexts"][0]["name"]
    with output_path.open("x") as stream:
        output_path.chmod(0o600)
        stream.write(yaml.safe_dump(merged))
    return {"kubeconfig": str(output_path), "credentials_exported": True}


def server_config(config):
    """Bind each private scheduler workspace to a personal worker/controller pair.

    Args:
        config: Validated installation configuration.
    Returns:
        Private SkyPilot placement and identity configuration.
    Raises:
        None.
    """
    contexts, workspaces = {}, {}
    for binding in allocations(config):
        worker, control = worker_context(binding), controller_context(binding)
        contexts[worker] = {
            "remote_identity": WORKER_ACCOUNT,
            "pod_config": {"spec": {"automountServiceAccountToken": False}},
        }
        contexts[control] = {
            "remote_identity": CONTROLLER_ACCOUNT,
            "post_provision_runcmd": [_controller_bootstrap(binding)],
        }
        workspaces[scheduler_workspace(binding)] = {
            "kubernetes": {"allowed_contexts": [worker, control]},
        }
    return {
        "rbac": {"default_role": "user"},
        "jobs": {"controller": {"consolidation_mode": False}},
        "allowed_clouds": ["kubernetes"],
        "kubernetes": {
            "allowed_contexts": list(contexts),
            "networking": "portforward",
            "context_configs": contexts,
        },
        "workspaces": workspaces,
    }


def scheduler_user(binding):
    """Keep each person's cluster controller separate while the ledger retains identity.

    Args:
        binding: Personal workspace and cluster allocation.
    Returns:
        Stable scheduler identity without external subject values.
    Raises:
        None.
    """
    identity = f"{binding.namespace}/{binding.cluster}"
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def scheduler_workspace(binding):
    """Keep internal scheduler workspace names within Kubernetes label length limits.

    Args:
        binding: Authorized personal cluster allocation.
    Returns:
        Stable internal workspace name without external identity values.
    Raises:
        None.
    """
    return "team-" + scheduler_user(binding)


def _append_contexts(merged, source, binding):
    for context_name, namespace in (
        (worker_context(binding), binding.namespace),
        (controller_context(binding), binding.namespace + "-control"),
    ):
        cluster, user = (
            copy.deepcopy(source["clusters"][0]),
            copy.deepcopy(source["users"][0]),
        )
        cluster["name"], user["name"] = context_name, context_name
        merged["clusters"].append(cluster)
        merged["users"].append(user)
        merged["contexts"].append(
            {
                "name": context_name,
                "context": {
                    "cluster": context_name,
                    "user": context_name,
                    "namespace": namespace,
                },
            }
        )


def _controller_bootstrap(binding):
    worker, control = worker_context(binding), controller_context(binding)
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": worker,
        "clusters": [_controller_cluster(binding)],
        "users": [
            {
                "name": "controller",
                "user": {
                    "tokenFile": "/var/run/secrets/kubernetes.io/serviceaccount/token",
                },
            }
        ],
        "contexts": [
            {
                "name": name,
                "context": {
                    "cluster": "local",
                    "user": "controller",
                    "namespace": namespace,
                },
            }
            for name, namespace in (
                (worker, binding.namespace),
                (control, binding.namespace + "-control"),
            )
        ],
    }
    encoded = base64.b64encode(json.dumps(config).encode()).decode()
    return (
        "umask 077; mkdir -p ~/.kube; printf %s "
        + shlex.quote(encoded)
        + " | base64 -d > ~/.kube/config"
    )


def _controller_api_address(binding):
    address = ipaddress.ip_network(binding.connection.api_server_cidr).network_address
    host = f"[{address}]" if address.version == 6 else str(address)
    return f"https://{host}:{binding.connection.api_server_port}"


def _controller_cluster(binding):
    return {
        "name": "local",
        "cluster": {
            "server": _controller_api_address(binding),
            "tls-server-name": urlsplit(binding.connection.api_server_url).hostname,
            "certificate-authority": "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
        },
    }


def _write(path, content):
    path.touch(mode=0o600, exist_ok=False)
    path.write_text(content)
