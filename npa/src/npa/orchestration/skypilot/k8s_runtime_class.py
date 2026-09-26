"""Verify explicitly requested Kubernetes RuntimeClasses before launch."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Callable, Mapping, Sequence

from npa.execution_preflight import ExecutionPreflightError


def _runtime_class_name(
    pod_spec: Mapping[str, Any], inherited: str | None
) -> str | None:
    if "runtimeClassName" not in pod_spec:
        return inherited
    name = pod_spec.get("runtimeClassName")
    if name is None or name == "":
        return None
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ExecutionPreflightError(
            "runtime_class",
            "runtimeClassName must be a resolved string, null, or empty",
        )
    return name


def _effective_names(
    task_pod_specs: Sequence[Mapping[str, Any]],
    global_pod_spec: Mapping[str, Any],
) -> tuple[str, ...]:
    # SkyPilot applies task cluster_config_overrides over global config. The
    # presence of the task key therefore matters even when it selects default.
    inherited = _runtime_class_name(global_pod_spec, None)
    requested: set[str] = set()
    for spec in task_pod_specs:
        name = _runtime_class_name(spec, inherited)
        if name is not None:
            requested.add(name)
    return tuple(sorted(requested))


def _raise_lookup_failure(name: str, result: subprocess.CompletedProcess[str]) -> None:
    output = f"{result.stderr or ''}\n{result.stdout or ''}".casefold()
    if "notfound" in output or "not found" in output:
        raise ExecutionPreflightError(
            "runtime_class",
            f'configured runtimeClassName "{name}" is not installed in the selected cluster; '
            "remove the explicit runtimeClassName or select a cluster that provides it",
        )
    if "forbidden" in output or "cannot get resource" in output:
        raise ExecutionPreflightError(
            "runtime_class",
            "Kubernetes RBAC denied RuntimeClass inspection in the selected context",
            status="unknown",
        )
    if "unauthorized" in output or "authentication" in output:
        raise ExecutionPreflightError(
            "runtime_class",
            "Kubernetes authentication failed while inspecting RuntimeClasses in the selected context",
            status="unknown",
        )
    raise ExecutionPreflightError(
        "runtime_class",
        "kubectl could not verify RuntimeClasses in the selected context",
        status="unknown",
    )


def _verify_lookup_payload(name: str, raw_payload: str) -> None:
    try:
        payload = json.loads(raw_payload or "{}")
    except (TypeError, ValueError) as exc:
        raise ExecutionPreflightError(
            "runtime_class",
            "kubectl returned unreadable RuntimeClass evidence",
            status="unknown",
        ) from exc
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if not isinstance(metadata, Mapping) or metadata.get("name") != name:
        raise ExecutionPreflightError(
            "runtime_class",
            "kubectl returned mismatched RuntimeClass evidence",
            status="unknown",
        )


def _lookup_runtime_class(
    name: str,
    *,
    context: str,
    environment: Mapping[str, str] | None,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    command = [
        "kubectl",
        "--context",
        context,
        "get",
        "runtimeclass.node.k8s.io",
        name,
        "--output=json",
    ]
    try:
        result = runner(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=dict(environment) if environment is not None else None,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExecutionPreflightError(
            "runtime_class",
            "kubectl could not inspect RuntimeClasses in the selected context",
            status="unknown",
        ) from exc
    if result.returncode != 0:
        _raise_lookup_failure(name, result)
    _verify_lookup_payload(name, result.stdout)


def verify_kubernetes_runtime_classes(
    task_pod_specs: Sequence[Mapping[str, Any]],
    *,
    context: str,
    global_pod_spec: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> None:
    """Verify requested RuntimeClasses in the exact selected cluster.

    Args:
        task_pod_specs: Task Kubernetes pod specifications before global overlay.
        context: Exact Kubernetes context selected for submission.
        global_pod_spec: Global pod specification inherited by each task.
        environment: Environment carrying the selected kubeconfig, when set.
        runner: Optional subprocess boundary used by unit tests.

    Returns:
        None.

    Raises:
        ExecutionPreflightError: The class is missing or cannot be verified.
    """
    execute = runner or subprocess.run
    for name in _effective_names(task_pod_specs, global_pod_spec or {}):
        _lookup_runtime_class(
            name, context=context, environment=environment, runner=execute
        )
