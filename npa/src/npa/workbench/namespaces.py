"""Create or reuse Kubernetes namespaces and export private client contexts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from npa.clients.kube import run_kubectl
from npa.clients.kubernetes_namespace import validate_namespace


def namespace_manifests(namespace: str) -> list[dict[str, Any]]:
    """Build a namespace manifest without changing authentication or access.

    Args:
        namespace: Kubernetes namespace name.
    Returns:
        One Namespace manifest; no service accounts or RBAC resources.
    Raises:
        ValueError: The namespace name is invalid.
    """
    validate_namespace(namespace)
    return [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}}]


def _kubectl(
    args: list[str], context: str, kubeconfig: str, *, stdin: str | None = None
) -> str:
    if not context.strip():
        raise ValueError("an explicit Kubernetes context is required")
    result = run_kubectl(args, context=context, kubeconfig=kubeconfig, stdin=stdin)
    if result.returncode:
        # Credential plugins and submitted payloads can appear in kubectl errors.
        raise ValueError(
            f"namespace operation failed during kubectl {args[0]}; check access to the selected context"
        )
    return result.stdout


def apply_namespace(
    namespace: str, *, context: str, kubeconfig: str = "", dry_run: bool = False
) -> dict[str, Any]:
    """Create a missing namespace, or reuse an existing one without modifying it.

    Args:
        namespace: Kubernetes namespace name.
        context: Exact authenticated context.
        kubeconfig: Source kubeconfig, or kubectl's normal resolution.
        dry_run: Return the manifest without contacting Kubernetes.
    Returns:
        Namespace, context, and created/existing/planned status.
    Raises:
        ValueError: Invalid input or insufficient Kubernetes access.
    """
    manifests = namespace_manifests(namespace)
    if not context.strip():
        raise ValueError("an explicit Kubernetes context is required")
    result = {"namespace": namespace, "context": context}
    if dry_run:
        return {**result, "status": "planned", "manifests": manifests}
    existing = _kubectl(
        ["get", "namespace", namespace, "--ignore-not-found", "-o", "name"],
        context,
        kubeconfig,
    )
    if existing.strip():
        return {**result, "status": "existing"}
    _kubectl(["create", "-f", "-"], context, kubeconfig, stdin=json.dumps(manifests[0]))
    return {**result, "status": "created"}


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
    sky_config: Path | None = None,
) -> dict[str, str]:
    """Export namespace configuration using existing credentials and SkyPilot settings.

    Args:
        namespace: Existing namespace, including Kubernetes' default namespace.
        context: Exact source context; its name and cluster identity are retained.
        output_dir: New private directory for client configuration and runtime state.
        kubeconfig: Source kubeconfig, or kubectl's normal resolution.
        sky_config: Existing SkyPilot config; otherwise use its environment/default path.
    Returns:
        Private configuration paths; credential contents are never returned.
    Raises:
        ValueError: Invalid namespace, configuration, or namespace access.
        OSError: A source cannot be read or the destination cannot be created.
    """
    validate_namespace(namespace)
    _kubectl(["get", "namespace", namespace, "-o", "name"], context, kubeconfig)
    document = _namespace_document(namespace, context, kubeconfig)
    sky = _sky_document(context, sky_config)
    return _write_context_bundle(namespace, context, output_dir, document, sky)


def _namespace_document(namespace, context, kubeconfig):
    raw = _kubectl(
        ["config", "view", "--minify", "--flatten", "--raw", "-o", "json"],
        context,
        kubeconfig,
    )
    try:
        document = json.loads(raw)
        contexts = document["contexts"]
        if len(contexts) != 1 or contexts[0]["name"] != context:
            raise ValueError
        contexts[0]["context"]["namespace"] = namespace
        document["current-context"] = context
        return document
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("source kubeconfig did not resolve the exact context") from exc


def _sky_document(context: str, explicit: Path | None) -> dict:
    configured = os.environ.get("SKYPILOT_GLOBAL_CONFIG", "")
    source = explicit or Path(configured or "~/.sky/config.yaml").expanduser()
    document = {}
    if explicit is not None or configured or source.exists():
        try:
            document = yaml.safe_load(source.expanduser().read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError("source SkyPilot configuration is invalid YAML") from exc
    if not isinstance(document, dict):
        raise ValueError("source SkyPilot configuration must be a mapping")
    kubernetes = document.setdefault("kubernetes", {})
    if not isinstance(kubernetes, dict):
        raise ValueError("source SkyPilot kubernetes configuration must be a mapping")
    document["allowed_clouds"] = ["kubernetes"]
    kubernetes["allowed_contexts"] = [context]
    return document


def _write_context_bundle(namespace, context, output_dir, document, sky):
    output_dir = output_dir.expanduser().absolute()
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    _private_write(output_dir / "kubeconfig", yaml.safe_dump(document))
    _private_write(output_dir / "sky.yaml", yaml.safe_dump(sky))
    return {
        "namespace": namespace,
        "context": context,
        "kubeconfig": str(output_dir / "kubeconfig"),
        "sky_config": str(output_dir / "sky.yaml"),
        "runtime_dir": str(output_dir / "runtime"),
    }
