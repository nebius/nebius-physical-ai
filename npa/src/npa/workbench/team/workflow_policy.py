"""Bind workflow and rendered task placement to authenticated execution allocations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from .authorization import ExecutionBinding
from .errors import AuthorizationError, TeamError
from .manifests import WORKER_ACCOUNT
from .storage import authorize_uri

_RESOURCE_FIELDS = frozenset(
    {
        "cpus",
        "memory",
        "accelerators",
        "num_nodes",
        "image",
        "image_id",
        "disk_size",
        "cloud",
        "region",
        "kubernetes",
    }
)
_RESERVED_ENV = (
    "AWS_",
    "NEBIUS_",
    "KUBECONFIG",
    "SKYPILOT_",
    "NPA_TEAM_",
    "NPA_CONFIG_DIR",
)


def prepare_document(document: dict, binding: ExecutionBinding, run_id: str) -> dict:
    """Resolve workflow placement and output storage before invoking the normal engine.

    Args:
        document: Untrusted NPA workflow JSON.
        binding: Server-authorized namespace, identity, and storage.
        run_id: Server-issued run identity.
    Returns:
        A copied workflow with fixed placement and run storage.
    Raises:
        TeamError: The input attempts unsupported placement or credential overrides.
    """
    result = copy.deepcopy(document)
    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict) or not isinstance(result.get("config", {}), dict):
        raise TeamError("workflow metadata and configuration must be mappings")
    if "namespace" in metadata:
        raise AuthorizationError("team execution namespace is assigned by the server")
    config = result.setdefault("config", {})
    grant = binding.allocation.storage
    config.update(
        bucket=grant.bucket, prefix=f"{grant.prefix.strip('/')}/runs/{run_id}"
    )
    resources = result.setdefault("resources", {"default": {}})
    if not isinstance(resources, dict):
        raise TeamError("workflow resources must be a mapping")
    for profile in resources.values():
        _bind_profile(profile, binding)
    _check_default_environment(result.get("run", {}))
    states = result.get("states", {})
    if not isinstance(states, dict):
        raise TeamError("workflow states must be a mapping")
    for state in states.values():
        if isinstance(state, dict) and "namespace" in state:
            raise AuthorizationError("all stages use the authenticated run namespace")
    return result


def load_bound_spec(document: dict, binding: ExecutionBinding, run_id: str, root: Path):
    """Validate a bound workflow through the repository's canonical schema and parser.

    Args:
        document, binding, run_id: Submitted workflow and trusted allocation.
        root: Private server staging directory.
    Returns:
        Parsed NPA workflow and the normalized document.
    Raises:
        TeamError: Schema or policy validation fails.
    """
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.spec import load_spec

    prepared = prepare_document(document, binding, run_id)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / "workflow.json"
    path.write_text(json.dumps(prepared))
    path.chmod(0o600)
    try:
        return load_spec(path), prepared
    except (NpaWorkflowError, ValueError, TypeError) as exc:
        raise TeamError(
            "workflow does not satisfy the NPA team execution contract"
        ) from exc


def enforce_rendered_tasks(
    path: Path, binding: ExecutionBinding, credentials: dict
) -> str:
    """Recheck every rendered wave before it crosses the SkyPilot boundary.

    Args:
        path: Canonical renderer's wave YAML.
        binding: Authenticated execution allocation.
        credentials: Only this person's explicitly scoped workload credentials.
    Returns:
        Validated YAML with mandatory identity and environment bindings.
    Raises:
        TeamError: A rendered task requests unsupported placement or file access.
    """
    documents = list(yaml.safe_load_all(path.read_text()))
    for task in documents:
        if not isinstance(task, dict):
            raise TeamError("rendered workflow contains an invalid task")
        if "resources" not in task:
            if set(task) <= {"name", "execution"}:
                continue
            raise TeamError("rendered task is missing its enforced resource placement")
        if any(task.get(key) for key in ("workdir", "file_mounts", "service")):
            raise AuthorizationError(
                "team tasks cannot mount server files or start public services"
            )
        _check_task_config(task)
        _bind_profile(task["resources"], binding)
        task["envs"] = {**task.get("envs", {}), **credentials}
        task["config"] = _worker_identity()
    return yaml.safe_dump_all(documents, sort_keys=False)


def _worker_identity():
    return {
        "kubernetes": {
            "pod_config": {
                "spec": {
                    "serviceAccountName": WORKER_ACCOUNT,
                    "automountServiceAccountToken": False,
                }
            }
        }
    }


def verify_artifact_scopes(steps, binding: ExecutionBinding) -> None:
    """Authorize each resolved input and output before a workflow wave is executed.

    Args:
        steps: Canonical plan steps with resolved artifact locations.
        binding: Personal output and explicit shared-input allocation.
    Returns:
        None.
    Raises:
        AuthorizationError: Any declared artifact escapes its permitted scope.
    """
    for step in steps:
        for output in step.outputs:
            uri = output.get("uri", "") if isinstance(output, dict) else str(output)
            authorize_uri(uri, binding.allocation.storage)
        for item in step.inputs:
            uri = item.get("uri", "") if isinstance(item, dict) else str(item)
            authorize_uri(
                uri, binding.allocation.storage, binding.allocation.shared_inputs
            )


def _bind_profile(profile, binding):
    if not isinstance(profile, dict) or set(profile) - _RESOURCE_FIELDS:
        raise TeamError("team resource profile contains unsupported fields")
    if profile.get("cloud") not in (None, "k8s", "kubernetes"):
        raise AuthorizationError(
            "team workloads must use their enrolled Kubernetes cluster"
        )
    if profile.get("region") not in (None, worker_context(binding)):
        raise AuthorizationError("team workloads cannot override their cluster context")
    if profile.get("kubernetes"):
        raise AuthorizationError(
            "team pod configuration is managed by the administrator"
        )
    profile.update(cloud="kubernetes", region=worker_context(binding))


def _check_default_environment(defaults):
    if not isinstance(defaults, dict):
        raise TeamError("workflow run defaults must be a mapping")
    environment = defaults.get("env", defaults.get("envs", {}))
    if not isinstance(environment, dict):
        raise TeamError("workflow environment must be a mapping")
    if any(str(key).startswith(_RESERVED_ENV) for key in environment):
        raise AuthorizationError("workload credentials are assigned by the server")


def _check_task_config(task):
    config = task.get("config") or {}
    if set(config) - {"kubernetes"}:
        raise AuthorizationError("task attempts to override server configuration")
    kubernetes = config.get("kubernetes") or {}
    if kubernetes:
        raise AuthorizationError(
            "task attempts to override administrator pod configuration"
        )


def worker_context(binding: ExecutionBinding) -> str:
    """Name a server-managed personal workload context.

    Args:
        binding: Authorized execution allocation.
    Returns:
        Stable, distinct context name.
    Raises:
        None.
    """
    return f"{binding.namespace}-{binding.cluster}-worker"


def controller_context(binding: ExecutionBinding) -> str:
    """Name a server-managed personal controller context.

    Args:
        binding: Authorized execution allocation.
    Returns:
        Stable, distinct context name.
    Raises:
        None.
    """
    return f"{binding.namespace}-{binding.cluster}-control"
