"""Refresh Nebius Container Registry pull credentials before K8s sim2real jobs."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from collections.abc import Sequence
from typing import Any

from npa.clients.nebius_auth import mint_nebius_iam_token, strip_ambient_token_env


def mint_nebius_registry_token(*, nebius_cli: str = "nebius") -> str:
    """Return a short-lived IAM token for ``cr.*.nebius.cloud`` pulls.

    Delegates to the canonical :func:`npa.clients.nebius_auth.mint_nebius_iam_token`,
    which performs a fresh profile-scoped exchange first (ambient token stripped,
    so a stale/wrong-identity ``NEBIUS_IAM_TOKEN`` can't be re-embedded into the
    very pull secret this refresh exists to fix — the ``403`` / ``ErrImagePull``
    failure), and only falls back to an injected ``NEBIUS_IAM_TOKEN`` when the
    ``nebius`` CLI is unavailable/fails — the in-pod case (token injected, no CLI
    on PATH). Raises ``NebiusTokenError`` (a ``RuntimeError``) if no token can be
    obtained, which best-effort callers catch.
    """

    return mint_nebius_iam_token(nebius_cli=nebius_cli)


def _registry_server_from_image(image: str) -> str:
    ref = image.removeprefix("docker:").strip()
    if "/" not in ref:
        return ""
    host = ref.split("/", 1)[0]
    if "." in host or ":" in host or host == "localhost":
        return host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return ""


def docker_config_json(*, registry_servers: Sequence[str], token: str) -> dict[str, Any]:
    """Return a dockerconfigjson whose ``auths`` covers every given registry host.

    A pull secret holds exactly one dockerconfigjson and ``kubectl apply`` replaces
    it wholesale, so every host a run pulls from has to be in the same document. One
    IAM token authenticates all of them, since the token is identity-scoped rather
    than host-scoped.
    """

    username = "iam"
    auth = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
    return {
        "auths": {
            server: {"username": username, "password": token, "auth": auth}
            for server in dict.fromkeys(server for server in registry_servers if server)
        }
    }


def _normalize_registry_server(value: str) -> str:
    server = value.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if not server.startswith("cr.") or ".nebius.cloud" not in server:
        return ""
    return server


def ensure_nebius_registry_pull_secret(
    *,
    registry_server: str = "",
    registry_servers: Sequence[str] = (),
    secret_name: str = "npa-nebius-registry",
    namespace: str = "default",
    kubeconfig: str = "",
    k8s_context: str = "",
    nebius_cli: str = "nebius",
) -> None:
    """Apply a fresh docker-registry secret so orchestrator pulls do not 401.

    Pass every host a run pulls from in one call. Applying them one at a time would
    leave only the host from the last call, because each apply replaces the secret,
    and pods pulling from the others would 401 — the failure this refresh exists to
    prevent.
    """

    servers = [
        normalized
        for normalized in (
            _normalize_registry_server(value) for value in (registry_server, *registry_servers)
        )
        if normalized
    ]
    if not servers:
        return
    token = mint_nebius_registry_token(nebius_cli=nebius_cli)
    payload = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "kubernetes.io/dockerconfigjson",
        "data": {
            ".dockerconfigjson": base64.b64encode(
                json.dumps(docker_config_json(registry_servers=servers, token=token)).encode(
                    "utf-8"
                )
            ).decode("ascii")
        },
    }
    cmd = ["kubectl"]
    if k8s_context:
        cmd.extend(["--context", k8s_context])
    cmd.extend(["-n", namespace, "apply", "-f", "-"])
    # Strip any ambient/stale NEBIUS_IAM_TOKEN so kubectl's nebius exec-credential
    # plugin re-authenticates via the configured profile instead of failing with
    # "Invalid token" (otherwise the pull-secret apply silently no-ops).
    env = strip_ambient_token_env(os.environ)
    if kubeconfig:
        env["KUBECONFIG"] = kubeconfig
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # In-pod orchestrators frequently have no kubectl on PATH, which raises
        # FileNotFoundError. Callers treat this refresh as best-effort and catch
        # RuntimeError, so keep every expected failure inside that contract
        # instead of letting an OSError escape and kill the run.
        raise RuntimeError(
            f"failed to apply registry pull secret {secret_name}: {exc}"
        ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"failed to apply registry pull secret {secret_name}: {detail}")


def ensure_registry_pull_secret_for_images(
    *images: str,
    secret_name: str = "npa-nebius-registry",
    namespace: str = "default",
    kubeconfig: str = "",
    k8s_context: str = "",
) -> None:
    servers = [server for server in map(_registry_server_from_image, images) if server]
    if not servers:
        return
    ensure_nebius_registry_pull_secret(
        registry_servers=servers,
        secret_name=secret_name,
        namespace=namespace,
        kubeconfig=kubeconfig,
        k8s_context=k8s_context,
    )
