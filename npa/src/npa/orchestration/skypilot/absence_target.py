"""Validate an existing key-backed reader against the original Kubernetes target."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from npa.cluster.absent_evidence import pinned_bytes, require


def _selection(data: bytes, context: str) -> tuple[dict, dict, dict]:
    config = yaml.safe_load(data)
    contexts = [row["context"] for row in config["contexts"] if row["name"] == context]
    require(len(contexts) == 1, "Kubernetes context is missing or ambiguous")
    selected = contexts[0]
    clusters = [
        row["cluster"]
        for row in config["clusters"]
        if row["name"] == selected["cluster"]
    ]
    users = [row["user"] for row in config["users"] if row["name"] == selected["user"]]
    require(len(clusters) == len(users) == 1, "Kubernetes target is ambiguous")
    cluster = clusters[0]
    require(
        set(cluster) <= {"server", "certificate-authority-data"}
        and str(cluster.get("server", "")).startswith("https://")
        and bool(cluster.get("certificate-authority-data")),
        "Explicit TLS endpoint and embedded CA required",
    )
    return selected, cluster, users[0]


def validate_target(manifest: dict, scope: dict) -> dict:
    """Compare target bytes without invoking the original authentication helper.

    Args:
        manifest: Pinned original registration and two kubeconfig files.
        scope: Native context and controller namespace.
    Returns:
        The registered cluster identity for a fresh provider read.
    Raises:
        ValueError: The selected target or reader authority differs.
    """
    import json

    context = scope["context"]
    old = _selection(pinned_bytes(manifest["original_kubeconfig"]), context)
    new = _selection(pinned_bytes(manifest["reader_kubeconfig"]), context)
    require(old[:2] == new[:2], "Original endpoint, CA, or context selection changed")
    require(
        old[0].get("namespace", "default") == scope["controller"]["namespace"],
        "Original namespace changed",
    )
    _reader_exec(new[2], manifest["authority"])
    registration = json.loads(pinned_bytes(manifest["cluster_registration"]))
    require(registration["name"] == context, "Wrong registered cluster context")
    require(
        registration.get("endpoint") in ("", old[1]["server"]),
        "Registered endpoint contradicts original target",
    )
    registration = {**registration, "original_tls": old[1]}
    return registration


def _reader_exec(user: dict, authority: dict) -> None:
    _reader_config_path(authority)
    require(
        set(user) == {"exec"},
        "Only the supported explicit key-profile reader is covered",
    )
    command = user["exec"]
    require(
        command["command"] == authority["binary"]["path"]
        and command["args"]
        == [
            "mk8s",
            "v1",
            "cluster",
            "get-token",
            "--profile",
            authority["profile"],
            "--format",
            "json",
        ]
        and not command.get("env"),
        "Reader has an unsupported credential selector",
    )


def _reader_config_path(authority: dict) -> Path:
    path = Path(authority["config"]["path"])
    require(
        path.name == "config.yaml", "Reader requires the default config.yaml basename"
    )
    return path


def reader_environment(authority: dict) -> dict[str, str]:
    """Select a pinned existing service-account profile without ambient tokens.

    Args:
        authority: Existing provider config, executable and key-file digests.
    Returns:
        A child-only environment; the caller environment is unchanged.
    Raises:
        ValueError: The exact key-backed selector cannot be validated.
    """
    config_path = _reader_config_path(authority)
    config = yaml.safe_load(pinned_bytes(authority["config"]))
    profile = config["profiles"][authority["profile"]]
    require(
        profile.get("auth-type") == "service account", "Key-backed profile required"
    )
    paths = [
        profile[key]
        for key in ("private-key-file-path", "service-account-credentials-file-path")
        if profile.get(key)
    ]
    require(paths == [authority["key"]["path"]], "Explicit reader key changed")
    require(
        not any("token" in key or "metadata" in key for key in profile),
        "Unsupported profile token source",
    )
    pinned_bytes(authority["key"])
    pinned_bytes(authority["binary"])
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NEBIUS_", "NPA_NEBIUS_"))
    }
    environment.pop("NPA_REUSE_IAM_TOKEN", None)
    environment.update(
        NEBIUS_CONFIG_DIR=str(config_path.parent),
        NEBIUS_PROFILE=authority["profile"],
        AWS_EC2_METADATA_DISABLED="true",
    )
    return environment
