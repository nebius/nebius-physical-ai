"""Manage team namespaces, Kubernetes access bindings, and private client contexts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from npa.clients.kube import run_kubectl
from npa.clients.kubernetes_namespace import validate_namespace


_LABEL = "npa.nebius.com/namespace"
_MANAGER = "npa-namespace"
_SERVICE_ACCOUNT = "npa-workbench"


def _validate_managed_namespace(namespace: str) -> str:
    validate_namespace(namespace)
    if namespace == "default" or namespace.startswith(("kube-", "skypilot-")):
        raise ValueError("choose a dedicated team namespace, not a system namespace")
    return namespace


def _subjects(users: tuple[str, ...], groups: tuple[str, ...]) -> list[dict]:
    subjects = []
    for kind, names in (("User", users), ("Group", groups)):
        for name in sorted(set(names)):
            if (
                not name.strip()
                or name != name.strip()
                or any(ord(c) < 32 for c in name)
            ):
                raise ValueError(
                    "RBAC subjects must be nonempty names without control characters"
                )
            if name.startswith("system:"):
                raise ValueError("system identities cannot be team members")
            subjects.append(
                {"kind": kind, "name": name, "apiGroup": "rbac.authorization.k8s.io"}
            )
    return subjects


def _object(kind: str, name: str, namespace: str, **fields: Any) -> dict:
    api = (
        "v1"
        if kind in {"Namespace", "ServiceAccount"}
        else "rbac.authorization.k8s.io/v1"
    )
    metadata = {
        "name": name,
        "labels": {_LABEL: namespace, "app.kubernetes.io/managed-by": _MANAGER},
    }
    if kind in {"ServiceAccount", "RoleBinding"}:
        metadata["namespace"] = namespace
    return {"apiVersion": api, "kind": kind, "metadata": metadata, **fields}


def _discovery_rules(namespace: str) -> list[dict]:
    rules = []
    for api, resource in (
        ("", "nodes"),
        ("node.k8s.io", "runtimeclasses"),
        ("storage.k8s.io", "storageclasses"),
    ):
        rules.append(
            {
                "apiGroups": [api],
                "resources": [resource],
                "verbs": ["get", "list", "watch"],
            }
        )
    rules.append(
        {
            "apiGroups": [""],
            "resources": ["namespaces"],
            "resourceNames": [namespace, "kube-system"],
            "verbs": ["get"],
        }
    )
    return rules


def _binding(kind, name, namespace, role, subjects):
    reference = {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "ClusterRole",
        "name": role,
    }
    return _object(kind, name, namespace, roleRef=reference, subjects=subjects)


def namespace_manifests(
    namespace: str, *, users: tuple[str, ...] = (), groups: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Build namespace access for researchers and an existing SkyPilot identity.

    Args:
        namespace: Dedicated team namespace.
        users: Complete set of Kubernetes usernames with edit access.
        groups: Complete set of identity-provider groups with edit access.
    Returns:
        Namespace, service account, and explicit RBAC manifests.
    Raises:
        ValueError: A namespace or member name is unsafe or invalid.
    """
    _validate_managed_namespace(namespace)
    members = _subjects(users, groups)
    return _namespace_access_objects(namespace, members)


def _namespace_access_objects(namespace, members):
    account = {
        "kind": "ServiceAccount",
        "name": _SERVICE_ACCOUNT,
        "namespace": namespace,
    }
    discovery = "npa-discovery-" + hashlib.sha256(namespace.encode()).hexdigest()[:20]
    documents = [
        _object("Namespace", namespace, namespace),
        _object("ServiceAccount", _SERVICE_ACCOUNT, namespace),
    ]
    documents.append(
        _binding("RoleBinding", "npa-workbench-edit", namespace, "edit", [account])
    )
    documents.append(
        _binding("RoleBinding", "npa-researchers-edit", namespace, "edit", members)
    )
    documents.append(
        _object("ClusterRole", discovery, namespace, rules=_discovery_rules(namespace))
    )
    documents.append(
        _binding(
            "ClusterRoleBinding", discovery, namespace, discovery, [account, *members]
        )
    )
    return documents


def _kubectl(
    args: list[str], context: str, kubeconfig: str, *, stdin: str | None = None
) -> str:
    if not context.strip():
        raise ValueError("an explicit Kubernetes context is required")
    result = run_kubectl(args, context=context, kubeconfig=kubeconfig, stdin=stdin)
    if result.returncode:
        # kubectl errors can include submitted manifests or credential-plugin output.
        raise ValueError(
            f"namespace operation failed during kubectl {args[0]}; check access to the selected context"
        )
    return result.stdout


def _verify_owned_objects(manifests: list[dict], context: str, kubeconfig: str) -> None:
    for manifest in manifests:
        metadata = manifest["metadata"]
        args = [
            "get",
            manifest["kind"],
            metadata["name"],
            "--ignore-not-found",
            "-o",
            "json",
        ]
        if "namespace" in metadata:
            args += ["--namespace", metadata["namespace"]]
        raw = _kubectl(args, context, kubeconfig)
        if not raw.strip():
            continue
        existing = json.loads(raw)
        labels = existing.get("metadata", {}).get("labels", {})
        if any(labels.get(key) != value for key, value in metadata["labels"].items()):
            raise ValueError(f"refusing to adopt an unmanaged {manifest['kind']}")


def apply_namespace(
    namespace: str,
    *,
    context: str,
    kubeconfig: str = "",
    users: tuple[str, ...] = (),
    groups: tuple[str, ...] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create or reconcile one NPA-owned namespace and its complete member list.

    Args:
        namespace: Dedicated team namespace.
        context: Explicit administrator context.
        kubeconfig: Administrator kubeconfig, or kubectl's normal resolution.
        users: Complete set of researcher usernames.
        groups: Complete set of researcher groups.
        dry_run: Return manifests without reading or changing the cluster.
    Returns:
        Non-secret namespace settings, and manifests for a dry run.
    Raises:
        ValueError: Invalid input, foreign resources, RBAC denial, or apply conflict.
    """
    manifests = namespace_manifests(namespace, users=users, groups=groups)
    if not context.strip():
        raise ValueError("an explicit Kubernetes context is required")
    result = {
        "namespace": namespace,
        "context": context,
        "service_account": _SERVICE_ACCOUNT,
    }
    if dry_run:
        return {**result, "status": "planned", "manifests": manifests}
    _verify_owned_objects(manifests, context, kubeconfig)
    _apply_manifests(manifests, context, kubeconfig)
    return {**result, "status": "applied"}


def _apply_manifests(manifests, context, kubeconfig):
    payload = {"apiVersion": "v1", "kind": "List", "items": manifests}
    _kubectl(
        ["apply", "--server-side", "--field-manager", _MANAGER, "-f", "-"],
        context,
        kubeconfig,
        stdin=json.dumps(payload),
    )


def _private_write(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(content)


def write_namespace_context(
    namespace: str,
    *,
    context: str,
    output_dir: Path,
    kubeconfig: str = "",
) -> dict[str, str]:
    """Export a private namespace context using the caller's existing identity.

    Args:
        namespace: Dedicated team namespace.
        context: Explicit source context; its name and cluster identity are retained.
        output_dir: New directory for kubeconfig, SkyPilot config and runtime state.
        kubeconfig: Source kubeconfig, or kubectl's normal resolution.
    Returns:
        Paths to private configs. Credential contents are never returned.
    Raises:
        ValueError: Invalid context, namespace, or namespace access.
        OSError: The destination exists or cannot be written privately.
    """
    _validate_managed_namespace(namespace)
    _kubectl(["get", "namespace", namespace, "-o", "name"], context, kubeconfig)
    document = _namespace_document(namespace, context, kubeconfig)
    return _write_context_bundle(namespace, context, output_dir, document)


def _namespace_document(namespace, context, kubeconfig):
    raw = _kubectl(
        ["config", "view", "--minify", "--flatten", "--raw", "-o", "json"],
        context,
        kubeconfig,
    )
    document = json.loads(raw)
    if (
        len(document.get("contexts", [])) != 1
        or document["contexts"][0]["name"] != context
    ):
        raise ValueError("source kubeconfig did not resolve the exact context")
    document["contexts"][0]["context"]["namespace"] = namespace
    document["current-context"] = context
    return document


def _write_context_bundle(namespace, context, output_dir, document):
    output_dir = output_dir.expanduser().absolute()
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    _private_write(output_dir / "kubeconfig", yaml.safe_dump(document))
    sky = {
        "allowed_clouds": ["kubernetes"],
        "kubernetes": {
            "allowed_contexts": [context],
            "remote_identity": _SERVICE_ACCOUNT,
        },
    }
    _private_write(output_dir / "sky.yaml", yaml.safe_dump(sky))
    return {
        "namespace": namespace,
        "context": context,
        "kubeconfig": str(output_dir / "kubeconfig"),
        "sky_config": str(output_dir / "sky.yaml"),
        "runtime_dir": str(output_dir / "runtime"),
    }
