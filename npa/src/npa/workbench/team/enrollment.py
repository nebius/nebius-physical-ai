"""Render, apply, and verify administrator-owned cluster execution boundaries."""

from __future__ import annotations

import json
import subprocess

import yaml

from .errors import BackendError, ConflictError
from .manifests import (
    CONTROLLER_ACCOUNT,
    WORKER_ACCOUNT,
    admission_manifests,
    execution_manifests,
)


def kubectl(binding, *arguments, input=None):
    """Call Kubernetes using only the explicitly enrolled context and credential file.

    Args:
        binding: Trusted cluster allocation.
        arguments, input: Administrator-generated arguments and optional stdin.
    Returns:
        Captured process result; raw diagnostics never go to public clients.
    Raises:
        BackendError: kubectl cannot be executed.
    """
    command = [
        "kubectl",
        "--kubeconfig",
        str(binding.connection.kubeconfig),
        "--context",
        binding.connection.context,
        *arguments,
    ]
    try:
        return subprocess.run(
            command, input=input, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise BackendError("Kubernetes enrollment transport is unavailable") from exc


def apply_enrollment(binding):
    """Apply a personal boundary only into absent or already-owned namespaces.

    Args:
        binding: Administrator-selected personal allocation.
    Returns:
        Non-secret enrollment result after verification.
    Raises:
        ConflictError, BackendError: Namespace ownership or cluster policy fails.
    """
    for name in (binding.namespace, binding.namespace + "-control"):
        response = kubectl(
            binding, "get", "namespace", name, "--ignore-not-found", "-o", "json"
        )
        if response.returncode:
            raise BackendError("cannot verify namespace ownership")
        if response.stdout.strip():
            labels = json.loads(response.stdout)["metadata"].get("labels", {})
            if labels.get("npa.nebius.ai/team-execution") != binding.namespace:
                raise ConflictError(
                    "refusing to adopt a namespace outside this allocation"
                )
    documents = admission_manifests() + execution_manifests(binding)
    result = kubectl(binding, "apply", "-f", "-", input=yaml.safe_dump_all(documents))
    if result.returncode:
        raise BackendError(
            "enrollment apply failed; inspect the cluster before retrying"
        )
    verify_enrollment(binding)
    return {
        "namespace": binding.namespace,
        "cluster": binding.cluster,
        "status": "enrolled",
    }


def verify_enrollment(binding):
    """Require installed quotas, identities, admission policy, and network boundaries.

    Args:
        binding: Current administrator-owned allocation.
    Returns:
        None on verified enrollment.
    Raises:
        BackendError: Any required resource or explicit denial is missing.
    """
    for desired in admission_manifests() + execution_manifests(binding):
        metadata = desired["metadata"]
        arguments = ["get", desired["kind"], metadata["name"], "-o", "json"]
        if metadata.get("namespace"):
            arguments += ["--namespace", metadata["namespace"]]
        response = kubectl(binding, *arguments)
        if response.returncode:
            raise BackendError("team cluster enrollment is missing or unreadable")
        actual = json.loads(response.stdout)
        if not _contains(actual, desired):
            raise BackendError("team cluster enrollment differs from current policy")
        if desired["kind"] == "ValidatingAdmissionPolicy":
            _check_admission(actual)
    _check_denials(binding)


def _contains(actual, desired):
    if isinstance(desired, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in desired.items()
        )
    if isinstance(desired, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(desired)
            and all(
                _contains(left, right)
                for left, right in zip(actual, desired, strict=True)
            )
        )
    return actual == desired


def _check_admission(actual):
    status = actual.get("status", {})
    if status.get("observedGeneration") != actual["metadata"].get("generation"):
        raise BackendError("team admission policy has not been checked by Kubernetes")
    if "typeChecking" not in status or status["typeChecking"].get("expressionWarnings"):
        raise BackendError(
            "team admission policy did not pass Kubernetes expression checks"
        )


def _check_denials(binding):
    accounts = [
        (binding.namespace, WORKER_ACCOUNT),
        (binding.namespace + "-control", CONTROLLER_ACCOUNT),
    ]
    for namespace, account in accounts:
        subject = f"system:serviceaccount:{namespace}:{account}"
        for resource in ("clusterrolebindings", "clusterroles", "namespaces"):
            result = kubectl(
                binding, "auth", "can-i", "create", resource, "--as", subject
            )
            if result.stdout.strip() != "no" or result.returncode not in (0, 1):
                raise BackendError(
                    "team identity has unexpected cluster administration access"
                )
    result = kubectl(
        binding,
        "auth",
        "can-i",
        "create",
        "pods",
        "--namespace",
        binding.namespace,
        "--as",
        f"system:serviceaccount:{binding.namespace}:{WORKER_ACCOUNT}",
    )
    if result.stdout.strip() != "no" or result.returncode not in (0, 1):
        raise BackendError("worker identity unexpectedly permits workload creation")
