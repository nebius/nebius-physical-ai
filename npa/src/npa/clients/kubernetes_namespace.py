"""Resolve the namespace selected by a Kubernetes context without changing it."""

from __future__ import annotations

import json
import re

from npa.clients.kube import run_kubectl


def validate_namespace(namespace: str) -> str:
    """Validate a Kubernetes namespace as a DNS label.

    Args:
        namespace: The exact namespace name.
    Returns:
        The validated name.
    Raises:
        ValueError: The name is empty or is not a DNS label of at most 63 bytes.
    """
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", namespace):
        raise ValueError("namespace must be a lowercase DNS label of 1–63 characters")
    return namespace


def context_namespace(*, context: str = "", kubeconfig: str = "") -> str:
    """Read the effective context namespace, failing closed on unreadable config.

    Args:
        context: Exact context, or the selected context when omitted.
        kubeconfig: Explicit kubeconfig; otherwise use kubectl's normal resolution.
    Returns:
        The context namespace, or Kubernetes' default when its field is absent.
    Raises:
        ValueError: The context cannot be read or its namespace is invalid.
    """
    result = run_kubectl(
        ["config", "view", "--minify", "-o", "json"],
        context=context,
        kubeconfig=kubeconfig,
    )
    if result.returncode:
        raise ValueError(
            "cannot resolve namespace from the selected Kubernetes context"
        )
    try:
        contexts = json.loads(result.stdout)["contexts"]
        if len(contexts) != 1:
            raise ValueError
        selected = contexts[0]
        if context and selected["name"] != context:
            raise ValueError
        namespace = selected["context"].get("namespace") or "default"
        return validate_namespace(namespace)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "selected Kubernetes context has an invalid namespace"
        ) from exc
